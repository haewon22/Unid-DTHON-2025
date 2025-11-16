import os
import io
import json
import math
import random
import argparse
from glob import glob
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import shutil

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer, 
    AutoImageProcessor
)

from model import (
    MatcherDataset,
    MatcherModel,
    contrastive_loss,
    compute_alignment_loss,
    find_jsons,
    get_image_path
)

MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

TXT_ENCODER_NAME = "klue/roberta-base"
IMG_ENCODER_NAME = "microsoft/swin-base-patch4-window7-224" 
EMBED_DIM = 512 



def convert_to_yolo_format(json_files: List[str], output_dir: str, jpg_dir: str = None):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    corrupt_count = 0
    files_with_targets = 0
    for jf_path in tqdm(json_files, desc="Converting"):
        try:
            with open(jf_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            img_path = get_image_path(jf_path, data, jpg_dir) 
            annotations = data.get("learning_data_info", {}).get("annotation", [])
            if not annotations: continue
            W, H = data.get("source_data_info", {}).get("document_resolution", [1, 1])
            if W <= 0 or H <= 0: W, H = 1, 1
            lines = []
            for ann in annotations:
                bbox = ann.get("bounding_box")
                class_id = str(ann.get("class_id", ""))
                if not bbox or not class_id.startswith('V'):
                    continue
                x, y, w, h = bbox
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                if x2 <= x1 or y2 <= y1: continue
                w_clipped = x2 - x1
                h_clipped = y2 - y1
                cx = (x1 + w_clipped / 2.0) / W
                cy = (y1 + h_clipped / 2.0) / H
                nw = w_clipped / W
                nh = h_clipped / H
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= nw <= 1 and 0 <= nh <= 1):
                    corrupt_count += 1
                    continue
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            if lines:
                files_with_targets += 1
                img_basename = os.path.basename(img_path)
                img_dest = os.path.join(images_dir, img_basename)
                if not os.path.exists(img_dest):
                    shutil.copy2(img_path, img_dest)
                label_basename = os.path.splitext(img_basename)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_basename)
                with open(label_path, 'w') as lf:
                    lf.writelines(lines)
        except Exception as e:
            print(f"[Warning] Error converting {jf_path}: {e}")
    print(f"Conversion complete. Data saved to {output_dir}")
    print(f"  Processed {files_with_targets} images containing 'V' elements.")
    if corrupt_count > 0:
        print(f"[Warning] Skipped {corrupt_count} bounding boxes with out-of-bounds coordinates.")


def create_yaml_config(train_dir: str, val_dir: str, output_path: str):
    yaml_content = f"""# DocLayout-YOLO Training Configuration
path: {os.path.abspath(os.path.dirname(train_dir))}
train: {os.path.basename(train_dir)}/images
val: {os.path.basename(val_dir)}/images
# Classes
nc: 1
names: ['document_element']
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    print(f"YAML config saved to {output_path}")



def main_matcher(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BASE_PATH = args.base_path
    TRAIN_PATH = os.path.join(BASE_PATH, "train_valid/train")
    VALID_PATH = os.path.join(BASE_PATH, "train_valid/valid")
    
    tokenizer = AutoTokenizer.from_pretrained(TXT_ENCODER_NAME)
    image_processor = AutoImageProcessor.from_pretrained(IMG_ENCODER_NAME)
    
    train_press_jsons = find_jsons(os.path.join(TRAIN_PATH, "press_json")) 
    train_report_jsons = find_jsons(os.path.join(TRAIN_PATH, "report_json"))
    all_train_jsons = train_press_jsons + train_report_jsons
    
    val_press_jsons = find_jsons(os.path.join(VALID_PATH, "press_json"))
    val_report_jsons = find_jsons(os.path.join(VALID_PATH, "report_json"))
    all_val_jsons = val_press_jsons + val_report_jsons
    
    if args.sample_ratio < 1.0:
        random.shuffle(all_train_jsons)
        random.shuffle(all_val_jsons)
        all_train_jsons = all_train_jsons[:int(len(all_train_jsons) * args.sample_ratio)]
        all_val_jsons = all_val_jsons[:int(len(all_val_jsons) * args.sample_ratio)]

    train_dataset = MatcherDataset(
        json_files=all_train_jsons,
        tokenizer=tokenizer,
        image_processor=image_processor,
        jpg_dir=None,
        max_len=args.tx_max_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    print("Initializing Matcher Model (Swin-Base)...")
    model = MatcherModel( 
        txt_model_name=TXT_ENCODER_NAME,
        img_model_name=IMG_ENCODER_NAME,
        embed_dim=EMBED_DIM
    ).to(device)
    
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "text_encoder" in name or "image_encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.learning_rate},
            {"params": head_params, "lr": args.learning_rate * 10}
        ],
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    output_dir = "./checkpoints_matcher"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        is_warmup = epoch <= 3
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Train (Warmup={is_warmup})")
        
        for batch in train_loader_tqdm:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            positive_image = batch["positive_image"].to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                anchor_embed, positive_embed = model(input_ids, attention_mask, positive_image)
                
            loss = contrastive_loss(anchor_embed, positive_embed)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        
        print("-" * 50)
        print(f"Epoch {epoch} Summary:")
        print(f"  Mode: {'WARMUP' if is_warmup else 'HARD MINING'}")
        print(f"  Train Loss: {avg_train_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.2e})")
        print("-" * 50)
        
        if epoch % 5 == 0 or epoch == args.epochs:
            save_path = os.path.join(output_dir, f"matcher_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🎉 Matcher model saved to {save_path}")



def main_dla(args):
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for DLA (YOLO)")
    
    BASE_PATH = args.base_path
    TRAIN_PATH = os.path.join(BASE_PATH, "train_valid/train")
    VALID_PATH = os.path.join(BASE_PATH, "train_valid/valid")
    
    YOLO_DATA_DIR = os.path.join(BASE_PATH, "yolo_format")
    TRAIN_YOLO_DIR = os.path.join(YOLO_DATA_DIR, "train")
    VAL_YOLO_DIR = os.path.join(YOLO_DATA_DIR, "val")
    
    train_press_jsons = find_jsons(os.path.join(TRAIN_PATH, "press_json")) 
    train_report_jsons = find_jsons(os.path.join(TRAIN_PATH, "report_json"))
    all_train_jsons = train_press_jsons + train_report_jsons
    
    val_press_jsons = find_jsons(os.path.join(VALID_PATH, "press_json"))
    val_report_jsons = find_jsons(os.path.join(VALID_PATH, "report_json"))
    all_val_jsons = val_press_jsons + val_report_jsons
    
    if args.sample_ratio < 1.0:
        random.shuffle(all_train_jsons)
        random.shuffle(all_val_jsons)
        all_train_jsons = all_train_jsons[:int(len(all_train_jsons) * args.sample_ratio)]
        all_val_jsons = all_val_jsons[:int(len(all_val_jsons) * args.sample_ratio)]
    
    if not os.path.exists(TRAIN_YOLO_DIR) or not os.path.exists(VAL_YOLO_DIR):
        convert_to_yolo_format(all_train_jsons, TRAIN_YOLO_DIR)
        convert_to_yolo_format(all_val_jsons, VAL_YOLO_DIR)
        yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
        create_yaml_config(TRAIN_YOLO_DIR, VAL_YOLO_DIR, yaml_path)
    else:
        yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
    
    print(f"Loading DocLayout-YOLO model from {MODEL_REPO}...")
    if args.use_pretrained:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        model = YOLOv10(model_path)
    else:
        model = YOLOv10("yolov10n.pt")
    
    output_dir = "./runs/doclayout_yolo"
    os.makedirs(output_dir, exist_ok=True)
    
    model.train(
        data=yaml_path,
        epochs=args.epochs_dla,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device,
        lr0=args.lr_dla,
        lrf=args.lr_dla / 100,
        project=output_dir,
        name="dla_exp",
        exist_ok=True,
        pretrained=args.use_pretrained,
        optimizer='AdamW',
        verbose=True,
        val=True,
        save=True,
        save_period=args.save_period,
        patience=args.patience,
        workers=args.num_workers,
        cos_lr=True,
        close_mosaic=10,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLA (Stage 1) + Matcher (Stage 2) Training")
    
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--base_path", type=str, default="/root")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_ratio", type=float, default=0.1)

    parser.add_argument("--skip_dla", action="store_true")
    parser.add_argument("--epochs_dla", type=int, default=15)
    parser.add_argument("--batch_size_dla", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--lr_dla", type=float, default=1e-3)
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--save_period", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--skip_matcher", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--tx_max_len", type=int, default=64)

    args = parser.parse_args()
    
    if not args.skip_dla:
        print("\n" + "="*50)
        print("🚀 STARTING STAGE 1: DLA (YOLO) TRAINING")
        print("="*50)
        dla_args = argparse.Namespace(**vars(args))
        dla_args.batch_size = args.batch_size_dla 
        main_dla(dla_args)
    else:
        print("⏩ Skipping Stage 1: DLA (YOLO) Training.")

    if not args.skip_matcher:
        print("\n" + "="*50)
        print("🚀 STARTING STAGE 2: MATCHER (RANKER) TRAINING")
        print("="*50)
        main_matcher(args)
    else:
        print("⏩ Skipping Stage 2: Matcher (Ranker) Training.")
