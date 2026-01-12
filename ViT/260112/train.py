import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import SwinForImageClassification, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import Optional


# =========================
# 설정 (임시 학습 최소 버전)

# 학습 재현성 SEED: 랜덤 고정(결과 재현)

# 모델 MODEL_ID: backbone 바꿈
# tiny → base로 갈수록 성능↑ / 속도↓ / VRAM↑

# 입력 크기 IMG_SIZE: 224/256/384 가능(모델/리소스에 따라)
# 크게 하면 성능 좋아질 수도 있지만 느려짐.

# 비디오 프레임 정책
# NUM_FRAMES: 비디오에서 뽑는 프레임 수(균등 샘플링 방식일 때)

# 학습량
# EPOCHS: 학습 횟수 (임시 2~3, 제대로면 10~20)
# BATCH_SIZE: 크면 빠르지만 VRAM 많이 먹음

# 최적화
# LR, WEIGHT_DECAY, WARMUP_EPOCHS, GRAD_CLIP
# 보통 LR=3e-4(헤드만 학습) / 전체 파인튜닝이면 더 낮게(예: 1e-5~1e-4)

# 프레임 로짓 집계
# AGG: mean / max / topkmean

# =========================
SEED = 42
MODEL_ID = "microsoft/swin-tiny-patch4-window7-224"  # 빠르게: tiny 추천
IMG_SIZE = 224
NUM_FRAMES = 16               # 임시학습이면 5로 줄이는게 빠름
EPOCHS = 8                   # 임시학습 3 추천
BATCH_SIZE = 8               # GPU 상황에 맞게 8/4/2
LR = 1e-4                    # 임시학습이면 3e-4 추천  
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0

AGG = "topkmean"             # mean|max|topkmean
TOPK = 2                     # 임시면 2~3

TRAIN_ROOT = Path("./train_data")
TEST_ROOT = Path("./test_data")

MODEL_OUT = Path("./model/model.pt")
OUT_DIR = Path("./output")
OUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_OUT.parent.mkdir(exist_ok=True, parents=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 속도 더 원하면 cudnn.benchmark=True로 두는데, 그럼 재현성이 약해짐.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 비디오/이미지 읽기
# 프레임 추출 블록 — “비디오를 어떻게 이미지 묶음으로 바꿀지”
# A) uniform_frame_indices()
# 균등하게 num_frames개 뽑는 인덱스를 만들어줌.
# B) read_rgb_frames_every_2s()
# FPS를 읽어서 2초당 1프레임 뽑음
# max_frames로 길이 제한 가능
# C) read_rgb_frames()
# 이미지면 1장 반환
# 비디오면 “어떤 정책으로 프레임을 뽑을지”가 들어가는 곳
# =========================
def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def read_rgb_frames_every_2s(
    file_path: Path,
    max_frames: Optional[int] = None
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(file_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total <= 0:
        cap.release()
        return []

    # FPS가 0/NaN으로 나오는 파일도 있어서 방어
    if fps is None or fps <= 1e-6:
        # 대충 30fps 가정(최악의 fallback) — 가능하면 메타데이터로 보정하는 게 좋음
        fps = 30.0

    step = max(1, int(round(fps * 2.0)))  # 2초당 1프레임 -> fps*(간격) 1이면 1프레임
    idxs = list(range(0, total, step))

    # 너무 길면 최대 프레임 제한(선택)
    if max_frames is not None and len(idxs) > max_frames:
        # 균등하게 줄이기
        idxs = np.linspace(0, total - 1, max_frames, dtype=int).tolist()

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames



# def read_rgb_frames(file_path: Path, num_frames: int = NUM_FRAMES) -> list[np.ndarray]:
#     ext = file_path.suffix.lower()

#     if ext in IMAGE_EXTS:
#         from PIL import Image
#         img = Image.open(file_path).convert("RGB")
#         return [np.array(img)]

#     if ext in VIDEO_EXTS:
#         return read_rgb_frames_every_2s(file_path, max_frames=num_frames)

#     return []

def read_rgb_frames(file_path: Path, num_frames: int = NUM_FRAMES) -> list[np.ndarray]:
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTS:
        from PIL import Image
        img = Image.open(file_path).convert("RGB")
        return [np.array(img)]

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        idxs = uniform_frame_indices(total, num_frames)

        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    return []




# =========================
# Transform (Swin용)
# Transform 블록 — “데이터 증강/정규화”

# train_transform은 학습용(증강 포함)

# val_transform은 검증용(증강 X)
# RandomCrop → RandomResizedCrop(더 강한 증강)
# Normalize를 모델 표준(ImageNet mean/std)로 바꾸기
# =========================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])


# =========================
# 집계 (프레임 -> 파일 1개 로짓)
# =========================
def aggregate_logits(logits: torch.Tensor, method="mean", topk=3) -> torch.Tensor:
    # logits: [T]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    method = method.lower()
    if method == "mean":
        return logits.mean()
    if method == "max":
        return logits.max()
    if method in ("topkmean", "topk"):
        k = max(1, min(int(topk), logits.numel()))
        return torch.topk(logits, k=k).values.mean()
    raise ValueError(f"Unknown agg method: {method}")


# =========================
# train.csv / val.csv 자동 생성 (폴더 분리형)
# =========================
def make_train_val_csv(root: Path, out_train="train.csv", out_val="val.csv", test_size=0.2):
    real_dir = root / "real"
    fake_dir = root / "fake"

    rows = []
    for p in real_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS):
            rows.append({"path": str(p.relative_to(root)).replace("\\", "/"), "label": 0})
    for p in fake_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS):
            rows.append({"path": str(p.relative_to(root)).replace("\\", "/"), "label": 1})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("train_data/real, train_data/fake 아래에 학습 파일이 0개입니다.")

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df["label"]
    )
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    print(f"[CSV] train={len(train_df)} val={len(val_df)} saved -> {out_train}, {out_val}")


# =========================
# Dataset
# =========================
class MediaDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: Path, transform, num_frames=NUM_FRAMES):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel = self.df.iloc[idx]["path"]
        y = float(self.df.iloc[idx]["label"])
        fp = self.root / rel

        frames = read_rgb_frames(fp, num_frames=self.num_frames)

        # 프레임이 없으면 더미 1장
        if len(frames) == 0:
            xs = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        else:
            tensors = [self.transform(f) for f in frames]  # each [3,224,224]
            xs = torch.stack(tensors, dim=0)              # [T,3,224,224]

        return xs, torch.tensor([y], dtype=torch.float32)


def collate_media(batch):
    xs_list, ys_list = zip(*batch)
    return list(xs_list), torch.cat(ys_list, dim=0)  # xs_list: List[[T,3,224,224]]


# =========================
# forward helper
# forward_logits() — “프레임 묶음을 모델에 넣고 [T] 로짓으로”
# model(xs)에서 xs는 [T,3,224,224]라서 T개 프레임을 한꺼번에 처리
# 모델은 이를 배치 T로 처리
# 출력은 [T,1] 같은 형태 → squeeze → [T]로 정리

# =========================
def forward_logits(model, xs: torch.Tensor) -> torch.Tensor:
    # xs: [T,3,224,224]
    outputs = model(xs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    logits = logits.squeeze()

    # logits -> [T]
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 1:
        logits = logits.view(-1)

    return logits

# eval_auc() — 검증
# 파일 단위로:
# 프레임 로짓 [T]
# 집계 스칼라 logit
# sigmoid로 확률
# 전체 모아서 ROC-AUC 계산
@torch.no_grad()
def eval_auc(model, loader, loss_fn):
    model.eval()
    ys, ps = [], []
    losses = []

    for xs_list, y in loader:
        y = y.to(DEVICE)
        for xs, yi in zip(xs_list, y):
            xs = xs.to(DEVICE)
            logits_t = forward_logits(model, xs)               # [T]
            logit = aggregate_logits(logits_t, AGG, TOPK)      # scalar
            loss = loss_fn(logit.view(1), yi.view(1))
            prob = torch.sigmoid(logit).item()

            losses.append(float(loss.item()))
            ps.append(float(prob))
            ys.append(float(yi.item()))

    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return float(np.mean(losses)), float(auc)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn):
    model.train()
    losses = []

    for xs_list, y in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        y = y.to(DEVICE)

        batch_loss = 0.0

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            # 배치 내 파일 단위로 순회(가변 T 처리용)
            for xs, yi in zip(xs_list, y):
                xs = xs.to(DEVICE)
                logits_t = forward_logits(model, xs)
                logit = aggregate_logits(logits_t, AGG, TOPK)
                loss = loss_fn(logit.view(1), yi.view(1))
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / len(xs_list)

        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(float(batch_loss.item()))

    return float(np.mean(losses))



# 임시 학습(빠른 weight 만들기): 지금처럼 classifier만
# 성능 올리기:
# 마지막 stage만 풀기
# 전체 파인튜닝
# LR도 낮추기(예: 1e-5~1e-4)

def main():
    set_seed(SEED)
    print("Device:", DEVICE)
    
    # 1) CSV 생성(없으면 생성)
    if not Path("train.csv").exists() or not Path("val.csv").exists():
        make_train_val_csv(TRAIN_ROOT, "train.csv", "val.csv", test_size=0.2)

    # 2) 모델 로드 (binary head)
    print("Loading Swin:", MODEL_ID)
    model = SwinForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=1, #swin load
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # ✅ 임시학습 최소: backbone freeze(헤드만 학습) -> 빠르게 model.pt 만들기 좋음
    # 0) 전부 freeze
    for p in model.parameters():
        p.requires_grad = False

    # 1) classifier는 무조건 학습
    for p in model.classifier.parameters():
        p.requires_grad = True

    # 2) (선택) 마지막 stage도 학습
    if hasattr(model.swin, "encoder") and hasattr(model.swin.encoder, "layers"):
        for p in model.swin.encoder.layers[-1].parameters():
            p.requires_grad = True
    elif hasattr(model.swin, "layers"):
        for p in model.swin.layers[-1].parameters():
            p.requires_grad = True
    else:
        print("Swin 마지막 stage 경로를 못 찾음. print(model)로 구조 확인 필요")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    # 3) Dataset / Loader
    train_ds = MediaDataset("train.csv", TRAIN_ROOT, train_transform, num_frames=NUM_FRAMES)
    val_ds   = MediaDataset("val.csv",   TRAIN_ROOT, val_transform,   num_frames=NUM_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate_media)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, collate_fn=collate_media)

    # 4) Optim / Loss / Sched
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # 5) Train
    best_auc = -1.0
    best_path = MODEL_OUT

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn)
        va_loss, va_auc = eval_auc(model, val_loader, loss_fn)

        print(f"[{ep}/{EPOCHS}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_auc={va_auc:.4f}")

        # best 저장
        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), best_path)
            print("✅ saved best ->", best_path)

    # AUC가 nan(검증셋 한 클래스만)이어도 최소 저장은 해둠
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
        print("✅ saved(last) ->", best_path)

    print("Done. Final weight:", best_path)


if __name__ == "__main__":
    main()
