import os
import io
import json
import math
import random
import argparse
from glob import glob
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from functools import partial
import shutil 

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoImageProcessor
)

MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

TXT_ENCODER_NAME = "klue/roberta-base"
IMG_ENCODER_NAME = "microsoft/swin-base-patch4-window7-224" 
EMBED_DIM = 512 

def find_jsons(json_dir: str) -> List[str]:
    if os.path.isdir(json_dir):
        return sorted(glob(os.path.join(json_dir, "*.json")))
    print(f"[Warning] JSON 디렉터리를 찾을 수 없음: {json_dir}")
    return []

def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: str = None) -> str:
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)
    if not jpg_name:
        raise FileNotFoundError(f"No 'source_data_name_jpg' in {json_path}")
    if jpg_dir:
        path = os.path.join(jpg_dir, jpg_name)
        if os.path.exists(path):
            return path
    json_dir = os.path.dirname(json_path)
    if "_json" in json_dir:
        jpg_dir_guess = json_dir.replace("_json", "_jpg")
        path_guess = os.path.join(jpg_dir_guess, jpg_name)
        if os.path.exists(path_guess):
            return path_guess
    if "json" in json_dir:
        jpg_dir_guess_2 = json_dir.replace("json", "jpg")
        path_guess_2 = os.path.join(jpg_dir_guess_2, jpg_name)
        if os.path.exists(path_guess_2):
            return path_guess_2
    raise FileNotFoundError(f"Could not resolve JPG for {json_path} (jpg_name={jpg_name})")

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

class MatcherDataset(Dataset):
    def __init__(self, json_files: List[str], tokenizer, image_processor, jpg_dir: str = None, max_len: int = 64):
        self.json_files = json_files
        self.jpg_dir = jpg_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.items, self.img_to_anns = self._build_items()

    def _build_items(self):
        items = [] 
        img_to_anns = {} 
        print("Building Matcher dataset items...")
        for jf_path in tqdm(self.json_files, desc="Loading JSONs"):
            try:
                with open(jf_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                img_path = get_image_path(jf_path, data, self.jpg_dir) 
                annotations = data.get("learning_data_info", {}).get("annotation", [])
                if img_path not in img_to_anns:
                    img_to_anns[img_path] = []
                for ann in annotations:
                    if ann.get("visual_instruction") and ann.get("bounding_box") and ann.get("class_name"):
                        img_to_anns[img_path].append(ann)
            except Exception as e:
                print(f"[Dataset Warning] Skipping {jf_path}: {e}")
        
        for img_path, anns in img_to_anns.items():
            if not anns: 
                continue
            for i in range(len(anns)):
                items.append((img_path, i))

        print(f"Loaded {len(items)} valid items (Anchor/Positive pairs).")
        return items, img_to_anns

    def __len__(self):
        return len(self.items)
    
    def _get_image(self, img_path):
        try:
            return Image.open(img_path).convert("RGB")
        except Exception as e:
            img_size = (
                self.image_processor.size['height'], 
                self.image_processor.size['width']
            )
            return Image.new("RGB", img_size, color="white")
            
    def _crop_image(self, img, bbox):
        x, y, w, h = bbox
        return img.crop((x, y, x + w, y + h))

    def __getitem__(self, idx: int):
        pos_img_path, pos_ann_idx = self.items[idx]
        positive_anns = self.img_to_anns[pos_img_path]
        if pos_ann_idx >= len(positive_anns): pos_ann_idx = 0 
        pos_ann = positive_anns[pos_ann_idx]
        
        anchor_text = f"{pos_ann['class_name']} {self.tokenizer.sep_token} {pos_ann['visual_instruction']}"
        
        pos_img = self._get_image(pos_img_path)
        pos_crop = self._crop_image(pos_img, pos_ann["bounding_box"])

        tokenized_text = self.tokenizer(
            anchor_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        pos_crop_processed = self.image_processor(pos_crop, return_tensors="pt").pixel_values.squeeze(0)
        
        return {
            "input_ids": tokenized_text.input_ids.squeeze(0),
            "attention_mask": tokenized_text.attention_mask.squeeze(0),
            "positive_image": pos_crop_processed,
        }

class MatcherModel(nn.Module):
    def __init__(self, txt_model_name, img_model_name, embed_dim):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained(txt_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        self.image_encoder = AutoModel.from_pretrained(img_model_name)
        image_dim = self.image_encoder.config.hidden_size
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, embed_dim)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.ReLU(),
            nn.Linear(image_dim, embed_dim)
        )
        
    def forward(self, input_ids, attention_mask, positive_image):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        anchor_embed = self.text_proj(text_feat)
        anchor_embed = F.normalize(anchor_embed, p=2, dim=1)

        img_outputs = self.image_encoder(pixel_values=positive_image)

        feat = img_outputs.last_hidden_state
        feat = feat.mean(dim=1)

        positive_embed = self.image_proj(feat)
        positive_embed = F.normalize(positive_embed, p=2, dim=1)

        return anchor_embed, positive_embed


def contrastive_loss(A, P, temperature=0.07):
    """
    InfoNCE 기반 멀티모달 contrastive loss
    batch 내 다른 sample을 negative로 자동 사용
    """
    batch_size = A.size(0)
    A = F.normalize(A, dim=1)
    P = F.normalize(P, dim=1)

    logits = A @ P.t() / temperature
    labels = torch.arange(batch_size).long().to(A.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    return (loss_i2t + loss_t2i) / 2


def compute_alignment_loss(A, P):
    """Warmup Phase 용 Alignment Loss"""
    similarity = F.cosine_similarity(A, P, dim=1)
    loss = 1.0 - similarity.mean()
    return loss

def main_matcher(args):
    """
    2단계 Matcher 모델을 학습시킵니다. (Swin + BatchHard + Curriculum)
    """
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BASE_PATH = args.base_path
    TRAIN_PATH = os.path.join(BASE_PATH, "train_valid/train")
    VALID_PATH = os.path.join(BASE_PATH, "train_valid/valid")
    
    print("Initializing Tokenizer and Image Processor...")
    tokenizer = AutoTokenizer.from_pretrained(TXT_ENCODER_NAME)
    image_processor = AutoImageProcessor.from_pretrained(IMG_ENCODER_NAME)
    
    train_press_jsons = find_jsons(os.path.join(TRAIN_PATH, "press_json")) 
    train_report_jsons = find_jsons(os.path.join(TRAIN_PATH, "report_json"))
    all_train_jsons = train_press_jsons + train_report_jsons
    
    val_press_jsons = find_jsons(os.path.join(VALID_PATH, "press_json"))
    val_report_jsons = find_jsons(os.path.join(VALID_PATH, "report_json"))
    all_val_jsons = val_press_jsons + val_report_jsons
    
    if args.sample_ratio < 1.0:
        print(f"--- [Sampling Active] Using only {args.sample_ratio * 100:.0f}% of the data ---")
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
            
    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": args.learning_rate},
        {"params": head_params, "lr": args.learning_rate * 10} 
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=1e-4)
    
    loss_fn_triplet = nn.TripletMarginLoss(margin=1.0) 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    print(f"Starting Matcher Training for {args.epochs} epochs...")
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
                anchor_embed, positive_embed = model(
                    input_ids, attention_mask, positive_image
                )
                
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
        print(f"--- [DLA Sampling] Using only {args.sample_ratio * 100:.0f}% of the data ---")
        random.shuffle(all_train_jsons)
        random.shuffle(all_val_jsons)
        all_train_jsons = all_train_jsons[:int(len(all_train_jsons) * args.sample_ratio)]
        all_val_jsons = all_val_jsons[:int(len(all_val_jsons) * args.sample_ratio)]
    
    if not os.path.exists(TRAIN_YOLO_DIR) or not os.path.exists(VAL_YOLO_DIR):
        print("Converting data to YOLO format for DLA...")
        convert_to_yolo_format(all_train_jsons, TRAIN_YOLO_DIR)
        convert_to_yolo_format(all_val_jsons, VAL_YOLO_DIR)
        
        yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
        create_yaml_config(TRAIN_YOLO_DIR, VAL_YOLO_DIR, yaml_path)
    else:
        print("YOLO format data already exists. Skipping conversion.")
        yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
    
    print(f"Loading DocLayout-YOLO model from {MODEL_REPO}...")
    if args.use_pretrained:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        model = YOLOv10(model_path)
    else:
        model = YOLOv10("yolov10n.pt")
    
    print(f"Starting DLA (YOLO) Training for {args.epochs_dla} epochs...")
    output_dir = "./runs/doclayout_yolo"
    os.makedirs(output_dir, exist_ok=True)
    
    results = model.train(
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
    print("DLA Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLA (Stage 1) + Matcher (Stage 2) Training")
    
    parser.add_argument("--gpu_id", type=int, default=0, help="사용할 GPU 번호")
    parser.add_argument("--base_path", type=str, default="/root", help="데이터셋 기본 경로")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="사용할 데이터 비율 (0.1 = 10%)")

    parser.add_argument("--skip_dla", action="store_true", help="1단계 DLA(YOLO) 학습을 건너뛰기")
    parser.add_argument("--epochs_dla", type=int, default=15, help="[DLA] 학습 epoch 수")
    parser.add_argument("--batch_size_dla", type=int, default=16, help="[DLA] 배치 사이즈")
    parser.add_argument("--img_size", type=int, default=1024, help="[DLA] 입력 이미지 크기")
    parser.add_argument("--lr_dla", type=float, default=1e-3, help="[DLA] 초기 학습률")
    parser.add_argument("--use_pretrained", action="store_true", help="[DLA] 사전학습된 DocLayout-YOLO 모델 사용")
    parser.add_argument("--save_period", type=int, default=5, help="[DLA] 모델 저장 주기 (epoch)")
    parser.add_argument("--patience", type=int, default=10, help="[DLA] Early stopping patience")

    parser.add_argument("--skip_matcher", action="store_true", help="2단계 Matcher 학습을 건너뛰기")
    parser.add_argument("--epochs", type=int, default=10, help="[Matcher] 학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=32, help="[Matcher] 배치 사이즈 (더 크게 가능)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="[Matcher] 백본 학습률")
    parser.add_argument("--tx_max_len", type=int, default=64, help="[Matcher] 텍스트 최대 길이")

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