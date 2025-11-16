import os
import json
from glob import glob
from typing import List, Dict, Any
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import AutoModel


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
        except Exception:
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
        if pos_ann_idx >= len(positive_anns):
            pos_ann_idx = 0
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
    batch_size = A.size(0)
    A = F.normalize(A, dim=1)
    P = F.normalize(P, dim=1)

    logits = A @ P.t() / temperature
    labels = torch.arange(batch_size).long().to(A.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)

    return (loss_i2t + loss_t2i) / 2


def compute_alignment_loss(A, P):
    similarity = F.cosine_similarity(A, P, dim=1)
    loss = 1.0 - similarity.mean()
    return loss
