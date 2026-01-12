# train_facecrop_haar.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path
from typing import List

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
from PIL import Image

# (안정화) torch/cv2 스레드 제한
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# =========================
# 설정
# =========================
SEED = 42
MODEL_ID = "microsoft/swin-tiny-patch4-window7-224"
IMG_SIZE = 224
NUM_FRAMES = 16
EPOCHS = 8
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0

AGG = "topkmean"   # mean|max|topkmean
TOPK = 2

TRAIN_ROOT = Path("./train_data")
TEST_ROOT = Path("./test_data")

MODEL_OUT = Path("./model/model_facecrop.pt")
OUT_DIR = Path("./output")
OUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_OUT.parent.mkdir(exist_ok=True, parents=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Face crop (OpenCV Haar)
# =========================
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_face_haar(
    img: Image.Image,
    margin: float = 0.35,         # 딥페이크 단서(턱/머리카락)까지 포함하려고 조금 넉넉히
    make_square: bool = True,
    fill_color=(0, 0, 0),
) -> Image.Image:
    """
    Haar로 얼굴 1개(가장 큰 박스) 크롭. 실패 시 원본 반환.
    """
    img = img.convert("RGB")
    w, h = img.size

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = HAAR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return img

    x, y, fw, fh = max(faces, key=lambda t: t[2] * t[3])
    x1, y1, x2, y2 = x, y, x + fw, y + fh

    mx = int(fw * margin)
    my = int(fh * margin)
    x1 -= mx; y1 -= my; x2 += mx; y2 += my

    if make_square:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = max(x2 - x1, y2 - y1) // 2
        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    if pad_l or pad_t or pad_r or pad_b:
        canvas = Image.new("RGB", (w + pad_l + pad_r, h + pad_t + pad_b), fill_color)
        canvas.paste(img, (pad_l, pad_t))
        x1 += pad_l; x2 += pad_l
        y1 += pad_t; y2 += pad_t
        return canvas.crop((x1, y1, x2, y2))

    x1 = _clamp(x1, 0, w); x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h); y2 = _clamp(y2, 0, h)
    if x2 <= x1 or y2 <= y1:
        return img

    return img.crop((x1, y1, x2, y2))

# =========================
# Frame reading
# =========================
def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)

def read_rgb_frames(file_path: Path, num_frames: int = NUM_FRAMES) -> List[np.ndarray]:
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTS:
        img = Image.open(file_path).convert("RGB")
        img = crop_face_haar(img)
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
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil = crop_face_haar(pil)
            frames.append(np.array(pil))
        cap.release()
        return frames

    return []

# =========================
# Transforms (얼굴 크롭 후에는 공간 crop 강하게 X)
# =========================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# =========================
# Aggregation
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
# CSV 만들기
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
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
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

# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    if not Path("train.csv").exists() or not Path("val.csv").exists():
        make_train_val_csv(TRAIN_ROOT, "train.csv", "val.csv", test_size=0.2)

    print("Loading Swin:", MODEL_ID)
    model = SwinForImageClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True
    ).to(DEVICE)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Train classifier
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Train last stage
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

    train_ds = MediaDataset("train.csv", TRAIN_ROOT, train_transform, num_frames=NUM_FRAMES)
    val_ds   = MediaDataset("val.csv",   TRAIN_ROOT, val_transform,   num_frames=NUM_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_media)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_media)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_auc = -1.0
    best_path = MODEL_OUT

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn)
        va_loss, va_auc = eval_auc(model, val_loader, loss_fn)

        print(f"[{ep}/{EPOCHS}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_auc={va_auc:.4f}")

        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), best_path)
            print("✅ saved best ->", best_path)

    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
        print("✅ saved(last) ->", best_path)

    print("Done. Final weight:", best_path)

if __name__ == "__main__":
    main()
