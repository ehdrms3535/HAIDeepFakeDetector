import os
import random
from pathlib import Path
from typing import Optional, List, Tuple

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


# =========================
# 설정
# =========================
SEED = 42

MODEL_ID = "microsoft/swin-tiny-patch4-window7-224"
IMG_SIZE = 224

# ✅ 학습/추론 프레임 수는 통일 추천 (추론이 5면 학습도 5로)
NUM_FRAMES = 5

EPOCHS = 8
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0

# ✅ 안정형 집계(먼저 mean으로 점수 확인)
AGG = "mean"          # mean|max|topkmean
TOPK = 2              # topkmean일 때만 사용

TRAIN_ROOT = Path("./train_data")
MODEL_OUT = Path("./model/model.pt")
MODEL_OUT.parent.mkdir(exist_ok=True, parents=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 재현성
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 유틸: 균등 프레임 인덱스
# =========================
def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


# =========================
# ✅ 비디오/이미지 읽기 (frame_count=0 fallback 포함)
# =========================
def read_rgb_frames(file_path: Path, num_frames: int = NUM_FRAMES, seed: int = SEED) -> List[np.ndarray]:
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTS:
        from PIL import Image
        img = Image.open(file_path).convert("RGB")
        return [np.array(img)]

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ---- fallback: frame_count가 0/깨진 경우 -> reservoir sampling으로 num_frames만 뽑기
        if total <= 0:
            rng = random.Random(seed + hash(str(file_path)) % 10_000_000)
            picked: List[Tuple[int, np.ndarray]] = []
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i < num_frames:
                    picked.append((i, frame))
                else:
                    j = rng.randint(0, i)
                    if j < num_frames:
                        picked[j] = (i, frame)
                i += 1
            cap.release()

            if len(picked) == 0:
                return []
            picked.sort(key=lambda x: x[0])
            frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in picked]
            return frames

        # ---- 정상: total frame_count로 균등 샘플
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
# ✅ Transform (ImageNet 정규화로 통일)
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # ✅ RandomCrop 대신 RandomResizedCrop 권장 (얼굴 잘림 완화)
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =========================
# 집계
# =========================
def aggregate_logits(logits: torch.Tensor, method="mean", topk=3) -> torch.Tensor:
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
# train/val csv 생성
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
        raise RuntimeError("train_data/real, train_data/fake 아래 학습 파일이 0개입니다.")

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

        if len(frames) == 0:
            xs = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        else:
            tensors = [self.transform(f) for f in frames]
            xs = torch.stack(tensors, dim=0)

        return xs, torch.tensor([y], dtype=torch.float32)


def collate_media(batch):
    xs_list, ys_list = zip(*batch)
    return list(xs_list), torch.cat(ys_list, dim=0)


# =========================
# forward helper
# =========================
def forward_logits(model, xs: torch.Tensor) -> torch.Tensor:
    outputs = model(xs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    logits = logits.squeeze()

    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 1:
        logits = logits.view(-1)

    return logits


@torch.no_grad()
def eval_auc(model, loader, loss_fn):
    model.eval()
    ys, ps = [], []
    losses = []

    for xs_list, y in loader:
        y = y.to(DEVICE)
        for xs, yi in zip(xs_list, y):
            xs = xs.to(DEVICE)
            logits_t = forward_logits(model, xs)
            logit = aggregate_logits(logits_t, AGG, TOPK)
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


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    # 1) CSV 생성
    if not Path("train.csv").exists() or not Path("val.csv").exists():
        make_train_val_csv(TRAIN_ROOT, "train.csv", "val.csv", test_size=0.2)

    # 2) 모델 로드
    print("Loading Swin:", MODEL_ID)
    model = SwinForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # 3) 임시학습: 대부분 freeze + classifier(+마지막 stage)만 학습
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    # 마지막 stage 일부만 풀기(있는 경우)
    if hasattr(model, "swin"):
        if hasattr(model.swin, "encoder") and hasattr(model.swin.encoder, "layers"):
            for p in model.swin.encoder.layers[-1].parameters():
                p.requires_grad = True
        elif hasattr(model.swin, "layers"):
            for p in model.swin.layers[-1].parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    # 4) DataLoader
    train_ds = MediaDataset("train.csv", TRAIN_ROOT, train_transform, num_frames=NUM_FRAMES)
    val_ds = MediaDataset("val.csv", TRAIN_ROOT, val_transform, num_frames=NUM_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_media)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_media)

    # 5) Optim / Loss / Sched
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # 6) Train
    best_auc = -1.0
    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn)
        va_loss, va_auc = eval_auc(model, val_loader, loss_fn)

        print(f"[{ep}/{EPOCHS}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_auc={va_auc:.4f}")

        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), MODEL_OUT)
            print("✅ saved best ->", MODEL_OUT)

    if not MODEL_OUT.exists():
        torch.save(model.state_dict(), MODEL_OUT)
        print("✅ saved(last) ->", MODEL_OUT)

    print("Done. Final weight:", MODEL_OUT)


if __name__ == "__main__":
    main()
