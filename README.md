# Uni-DTHON-2025: DocLayout-AI Pipeline

문서 이미지에서 **구성요소 탐지(DLA: YOLO 기반)** → **텍스트·이미지
매칭(Matcher: Swin+RoBERTa 기반)** 두 단계를 학습하는 파이프라인입니다.

## 구성요소

### 1. **DLA (Document Layout Analyzer) -- YOLOv10 기반**

-   JSON → YOLO 포맷 변환 자동 처리\
-   DocLayout-YOLO 사전학습 모델 사용 가능\
-   문서 내 특정 요소 박스 검출 학습

### 2. **Matcher (Text--Image Alignment)**

-   RoBERTa 텍스트 인코더 + Swin 이미지 인코더\
-   시각 영역을 crop 후 텍스트 정보와 매칭\
-   Contrastive Loss로 텍스트 ↔ 이미지 임베딩 정렬

## 데이터 구조

    base_path/
     └── train_valid/
          ├── train/
          │    ├── press_json/
          │    └── report_json/
          └── valid/
               ├── press_json/
               └── report_json/

## 주요 실행 옵션

  옵션               설명                             기본값
  ------------------ -------------------------------- --------
  --skip_dla         YOLO 단계 건너뛰기               False
  --skip_matcher     Matcher 단계 건너뛰기            False
  --sample_ratio     데이터 샘플링 비율               0.1
  --epochs_dla       YOLO 학습 epoch                  15
  --epochs           Matcher 학습 epoch               10
  --use_pretrained   DocLayout-YOLO Pretrained 사용   False

## 실행 방법

### 전체 파이프라인 학습

``` bash
python train.py --base_path /path/to/dataset
```

### YOLO만 학습

``` bash
python train.py --skip_matcher
```

### Matcher만 학습

``` bash
python train.py --skip_dla
```

## 출력 구조

    checkpoints_matcher/      # Matcher 모델 가중치
    runs/doclayout_yolo/      # YOLO 학습 로그 및 weight
    yolo_format/              # 생성된 YOLO 포맷 데이터

## 특징 요약

-   JSON 기반 문서 데이터 자동 처리\
-   YOLO 기반 레이아웃 검출 + Multimodal Matching 2-Stage 구조\
-   Contrastive Loss 기반 텍스트-이미지 정렬\
-   Swin / RoBERTa 등 HuggingFace 기반 백본 사용
