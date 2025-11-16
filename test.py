import os
import io
import json
import math
import random
import argparse
from glob import glob
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import pandas as pd

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

TXT_ENCODER_NAME = "klue/roberta-base"
IMG_ENCODER_NAME = "microsoft/swin-base-patch4-window7-224" 
EMBED_DIM = 512 


class MatcherModel(nn.Module):
    def __init__(self, txt_model_name, img_model_name, embed_dim):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained(txt_model_name)
        text_dim = self.text_encoder.config.hidden_size # 768
        
        self.image_encoder = AutoModel.from_pretrained(img_model_name)
        image_dim = self.image_encoder.config.hidden_size # 1024
        
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
        
    @torch.no_grad()
    def forward_text(self, input_ids, attention_mask):
        """텍스트 쿼리만 인코딩 (Anchor)"""
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        anchor_embed = self.text_proj(text_feat)
        return F.normalize(anchor_embed, p=2, dim=1)

    @torch.no_grad()
    def forward_image(self, pixel_values):
        """이미지 크롭 배치만 인코딩 (Candidates)"""
        img_outputs = self.image_encoder(pixel_values=pixel_values)
        feat = img_outputs.last_hidden_state
        feat = feat.mean(dim=1) # GAP -> (B, C)
        positive_embed = self.image_proj(feat)
        return F.normalize(positive_embed, p=2, dim=1) 


class ImageCache:
    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, img_path):
        if img_path not in self.cache:
            if len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[img_path] = Image.open(img_path).convert("RGB")
        return self.cache[img_path]


def find_jsons(json_dir: str) -> List[str]:
    if os.path.isdir(json_dir):
        return sorted(glob(os.path.join(json_dir, "*.json")))
    print(f"[Warning] JSON 디렉터리를 찾을 수 없음: {json_dir}")
    return []

def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: str) -> str:
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)
    if not jpg_name:
        raise FileNotFoundError(f"No 'source_data_name_jpg' in {json_path}")
    
    path = os.path.join(jpg_dir, jpg_name)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Could not resolve JPG at {path}")

def get_test_items(json_files: List[str], jpg_dir: str):
    items = []
    print("Building test items...")
    for jf_path in tqdm(json_files, desc="Loading Test JSONs"):
        try:
            with open(jf_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_path = get_image_path(jf_path, data, jpg_dir) 
            annotations = data.get("learning_data_info", {}).get("annotation", [])
            
            for ann in annotations:
                class_name = ann.get("class_name")
                instruction = ann.get("visual_instruction")
                class_id = str(ann.get("class_id", ""))

                if instruction and class_name and class_id.startswith('V'):
                    items.append({
                        "query_id": ann.get("instance_id"),
                        "img_path": img_path,
                        "class_name": str(class_name).strip(),
                        "visual_instruction": str(instruction).strip(),
                        "orig_res": data.get("source_data_info", {}).get("document_resolution", [1, 1])
                    })
        except Exception as e:
            print(f"[Dataset Warning] Skipping {jf_path}: {e}")
    print(f"Loaded {len(items)} valid test items.")
    return items

def crop_image_from_normalized_bbox(img: Image.Image, norm_bbox: List[float]) -> Image.Image:
    W, H = img.size
    cx, cy, w, h = norm_bbox
    
    w = max(0, w)
    h = max(0, h)

    nw = w * W
    nh = h * H
    x1 = (cx * W) - (nw / 2.0)
    y1 = (cy * H) - (nh / 2.0)
    x2 = x1 + nw
    y2 = y1 + nh
    
    if x2 < x1: x2 = x1
    if y2 < y1: y2 = y1

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    
    if x1 == x2: x2 = min(W, x1 + 1)
    if y1 == y2: y2 = min(H, y1 + 1)

    return img.crop((x1, y1, x2, y2))

def convert_norm_to_pixel(norm_bbox: List[float], res: List[int]) -> List[float]:
    W, H = res
    cx, cy, w, h = norm_bbox
    
    nw = w * W
    nh = h * H
    x = (cx * W) - (nw / 2.0)
    y = (cy * H) - (nh / 2.0)
    
    return [x, y, nw, nh]

@torch.no_grad()
def main_predict_optimized(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading Stage 1 (DLA-YOLO) model from: {args.dla_model_path}")
    dla_model = YOLOv10(args.dla_model_path)
    dla_model.to(device) 

    print(f"Loading Stage 2 (Matcher) model from: {args.matcher_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(TXT_ENCODER_NAME)
    image_processor = AutoImageProcessor.from_pretrained(IMG_ENCODER_NAME)
    
    matcher_model = MatcherModel(
        txt_model_name=TXT_ENCODER_NAME,
        img_model_name=IMG_ENCODER_NAME,
        embed_dim=EMBED_DIM
    )
    matcher_model.load_state_dict(torch.load(args.matcher_model_path, map_location=device))
    matcher_model.to(device)
    matcher_model.eval()

    test_json_files = find_jsons(args.test_json_dir)
    
    if args.sample_ratio < 1.0:
        print(f"--- [Sampling Active] Using only {args.sample_ratio * 100:.0f}% of the test data ---")
        random.shuffle(test_json_files)
        test_json_files = test_json_files[:int(len(test_json_files) * args.sample_ratio)]
        
    test_items = get_test_items(test_json_files, args.test_img_dir)
    
    items_by_image = {}
    for item in test_items:
        img_path = item["img_path"]
        if img_path not in items_by_image:
            items_by_image[img_path] = []
        items_by_image[img_path].append(item)

    image_cache = ImageCache(max_size=args.cache_size)
    submission_rows = []
    
    print(f"\nStarting prediction on {len(items_by_image)} unique images...")
    
    for img_path, items in tqdm(items_by_image.items(), desc="Predicting Images"):
        
        yolo_results = dla_model.predict(
            img_path, 
            device=device, 
            verbose=False, 
            imgsz=args.img_size,
            batch=1
        )
        candidate_bboxes_norm = yolo_results[0].boxes.xywhn.to(device)

        if len(candidate_bboxes_norm) == 0:
            for item in items:
                submission_rows.append({
                    "query_id": item['query_id'],
                    "query_text": item['visual_instruction'],
                    "pred_x": 0.0, "pred_y": 0.0, "pred_w": 0.0, "pred_h": 0.0
                })
            continue

        query_texts = [f"{it['class_name']} {tokenizer.sep_token} {it['visual_instruction']}" for it in items]
        
        query_vecs_list = []
        for i in range(0, len(query_texts), args.text_batch_size):
            text_batch = query_texts[i:i+args.text_batch_size]
            tokenized = tokenizer(
                text_batch, 
                return_tensors="pt", 
                max_length=args.tx_max_len, 
                padding=True, 
                truncation=True
            )
            query_vecs = matcher_model.forward_text(
                tokenized.input_ids.to(device),
                tokenized.attention_mask.to(device)
            )
            query_vecs_list.append(query_vecs)
        
        query_embeds = torch.cat(query_vecs_list, dim=0)
        
        img = image_cache.get(img_path)
        
        candidate_crops = []
        for c_bbox_norm in candidate_bboxes_norm.cpu().tolist():
            crop = crop_image_from_normalized_bbox(img, c_bbox_norm)
            candidate_crops.append(crop)
        
        candidate_vecs = []
        for i in range(0, len(candidate_crops), args.image_batch_size):
            crops_batch = candidate_crops[i:i+args.image_batch_size]
            pixel_values_batch = image_processor(crops_batch, return_tensors="pt").pixel_values.to(device)
            vecs = matcher_model.forward_image(pixel_values_batch)
            candidate_vecs.append(vecs)
        
        candidate_embeds = torch.cat(candidate_vecs, dim=0)
        
        scores_matrix = torch.matmul(query_embeds, candidate_embeds.T)
        best_candidate_indices = scores_matrix.argmax(dim=1)

        for i in range(len(items)):
            item = items[i]
            best_idx = best_candidate_indices[i].item()
            pred_bbox_norm = candidate_bboxes_norm[best_idx].cpu().tolist()
            pred_bbox_pixel = convert_norm_to_pixel(pred_bbox_norm, item['orig_res'])
            
            submission_rows.append({
                "query_id": item['query_id'],
                "query_text": item['visual_instruction'],
                "pred_x": pred_bbox_pixel[0],
                "pred_y": pred_bbox_pixel[1],
                "pred_w": pred_bbox_pixel[2],
                "pred_h": pred_bbox_pixel[3],
            })

    submission_df = pd.DataFrame(submission_rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_h", "pred_w"])
    submission_df.to_csv(args.output_csv, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 Prediction Complete! Submission file saved to:")
    print(f"  {args.output_csv}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-Stage (DLA + SwinMatcher) Prediction")
    
    parser.add_argument("--gpu_id", type=int, default=0, help="사용할 GPU 번호")
    parser.add_argument("--test_json_dir", type=str, required=True, help=".../open/test/query")
    parser.add_argument("--test_img_dir", type=str, required=True, help=".../open/test/images")

    parser.add_argument("--dla_model_path", type=str, required=True, help="1단계 DLA(YOLO) 모델 가중치 (.pt)")
    parser.add_argument("--matcher_model_path", type=str, required=True, help="2단계 Matcher 모델 가중치 (.pth)")

    parser.add_argument("--output_csv", type=str, default="./submission.csv")

    parser.add_argument("--img_size", type=int, default=1024, help="[DLA] 입력 이미지 크기")
    parser.add_argument("--dla_batch_size", type=int, default=16, help="[DLA] 예측 시 사용할 배치 사이즈")

    parser.add_argument("--text_batch_size", type=int, default=128, help="[Matcher] 텍스트 인코딩 배치")
    parser.add_argument("--image_batch_size", type=int, default=128, help="[Matcher] 이미지 크롭 인코딩 배치")
    parser.add_argument("--tx_max_len", type=int, default=64, help="[Matcher] 텍스트 최대 길이")

    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (사용 안 함)")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--cache_size", type=int, default=50)

    args = parser.parse_args()
    
    main_predict_optimized(args)
