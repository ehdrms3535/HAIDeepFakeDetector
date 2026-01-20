# train.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from transformers import SwinForImageClassification, get_cosine_schedule_with_warmup
from PIL import ImageFilter
from PIL import Image

# =========================
# 기본 설정 (필요시 argparse로 오버라이드)
# =========================
SEED = 42

MODEL_ID = "microsoft/swin-base-patch4-window7-224"
IMG_SIZE = 224

TRAIN_NUM_FRAMES = 8
INFER_NUM_FRAMES = 16

AGG = "topkmean"      # mean | max | topkmean
TOPK_RATIO = 0.15     # topkmean 상위 비율(0.1~0.2 추천)

EPOCHS = 10
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0

# ✅ 경로 (너 환경에 맞게 조정 가능)
TRAIN_ROOT = Path(r"C:\Users\Kim\HAI\your_submission\train_data")
TEST_ROOT  = Path(r"C:\Users\Kim\HAI\your_submission\test_data")

MODEL_OUT  = Path("./model/model.pt")
CKPT_PATH  = Path("./model/ckpt.pt")   # resume용 체크포인트
OUT_DIR    = Path("./output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

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
# 체크포인트 저장/로드 (resume)
# =========================
def save_ckpt(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_auc: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best_auc": float(best_auc),
        "rng": {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(ckpt, path)


def load_ckpt(path: Path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    best_auc = float(ckpt.get("best_auc", -1.0))

    rng = ckpt.get("rng", None)
    if rng is not None:
        try:
            random.setstate(rng["py"])
            np.random.set_state(rng["np"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass

    return epoch, best_auc


# =========================
# 얼굴 크롭 (HaarCascade)
# =========================
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_face_haar(rgb: np.ndarray, margin: float = 0.35) -> np.ndarray:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return rgb

    x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])

    cx = x + bw / 2
    cy = y + bh / 2
    side = max(bw, bh) * (1.0 + 2.0 * margin)

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    if pad_l or pad_t or pad_r or pad_b:
        rgb = cv2.copyMakeBorder(
            rgb, pad_t, pad_b, pad_l, pad_r,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        x1 += pad_l; x2 += pad_l
        y1 += pad_t; y2 += pad_t

    return rgb[y1:y2, x1:x2]


# =========================
# 프레임 인덱스 (결정적)
# =========================
def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def read_rgb_frames(file_path: Path, num_frames: int) -> List[np.ndarray]:
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTS:
        try:
            img = Image.open(file_path).convert("RGB")
            return [np.array(img)]
        except Exception:
            return []

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: List[np.ndarray] = []

        if total <= 0:
            tmp = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                tmp.append(frame)
            cap.release()

            if len(tmp) == 0:
                return []
            total = len(tmp)
            idxs = uniform_frame_indices(total, num_frames)
            for idx in idxs:
                bgr = tmp[int(idx)]
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return frames

        idxs = uniform_frame_indices(total, num_frames)
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
# Transform
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class RandomDownUp:
    def __init__(self, p=0.7, min_scale=0.45, max_scale=0.95):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, self.max_scale)
        nw, nh = max(1, int(w*s)), max(1, int(h*s))
        img2 = img.resize((nw, nh), Image.BILINEAR)
        return img2.resize((w, h), Image.BILINEAR)


class RandomBlur:
    def __init__(self, p=0.4, rmin=0.4, rmax=1.5):
        self.p = p
        self.rmin = rmin
        self.rmax = rmax

    def __call__(self, img):
        if random.random() > self.p:
            return img
        r = random.uniform(self.rmin, self.rmax)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    RandomDownUp(p=0.7, min_scale=0.45, max_scale=0.95),
    RandomBlur(p=0.4, rmin=0.4, rmax=1.5),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

infer_transform = val_transform  # TTA 금지


# =========================
# 로짓 집계
# =========================
def aggregate_logits(logits: torch.Tensor, method="topkmean", topk_ratio=0.15) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    method = method.lower()
    if method == "mean":
        return logits.mean()
    if method == "max":
        return logits.max()
    if method in ("topkmean", "topk"):
        k = max(1, int(np.ceil(float(topk_ratio) * logits.numel())))
        return torch.topk(logits, k=k).values.mean()
    raise ValueError(f"Unknown agg method: {method}")


# =========================
# CSV 생성: fake=1, real=0 (기존 유지)
# =========================
def make_train_val_csv(root: Path, out_train="train.csv", out_val="val.csv", test_size=0.2, use_ratio=0.5):
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

    if use_ratio < 1.0:
        df, _ = train_test_split(
            df,
            train_size=use_ratio,
            stratify=df["label"],
            random_state=SEED
        )

    if len(df) == 0:
        raise RuntimeError("train_data/real, train_data/fake 아래 학습 파일이 0개입니다.")

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df["label"])
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    print(f"[CSV] train={len(train_df)} val={len(val_df)} saved -> {out_train}, {out_val}")


# =========================
# Dataset (파일 1개 -> [T,3,H,W], label, filename)
# =========================
class MediaDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: Path, transform, num_frames: int, face_crop: bool = True):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.face_crop = face_crop

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
            proc = []
            for f in frames:
                if self.face_crop:
                    f = crop_face_haar(f, margin=0.35)
                proc.append(self.transform(f))
            xs = torch.stack(proc, dim=0)

        return xs, torch.tensor([y], dtype=torch.float32), Path(rel).name


def collate_media(batch):
    xs_list, ys_list, names = zip(*batch)
    return list(xs_list), torch.cat(ys_list, dim=0), list(names)


def forward_logits(model, xs: torch.Tensor) -> torch.Tensor:
    outputs = model(pixel_values=xs)
    logits = outputs.logits
    logits = logits.squeeze()
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 1:
        logits = logits.view(-1)
    return logits


@torch.no_grad()
def eval_auc(model, loader, loss_fn, agg_method: str, topk_ratio: float, save_pred_path: Optional[str] = None):
    model.eval()
    ys, ps, losses, fns = [], [], [], []

    for xs_list, y, names in loader:
        y = y.to(DEVICE)  # [B,1]
        for xs, yi, nm in zip(xs_list, y, names):
            xs = xs.to(DEVICE, non_blocking=True)
            yi = yi.view(1)

            logits_t = forward_logits(model, xs)        # [T]
            target_t = yi.expand_as(logits_t)           # [T]
            loss = loss_fn(logits_t, target_t)
            losses.append(float(loss.item()))

            agg_logit = aggregate_logits(logits_t, agg_method, topk_ratio)
            prob_fake = torch.sigmoid(agg_logit).item()
            ps.append(float(prob_fake))
            ys.append(float(yi.item()))
            fns.append(str(nm))

    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")

    if save_pred_path is not None:
        out = pd.DataFrame({"filename": fns, "pred": ps, "label": ys})
        out.to_csv(save_pred_path, index=False)
        print("saved preds:", save_pred_path)

    return float(np.mean(losses)), float(auc)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn):
    model.train()
    losses = []

    for xs_list, y, _names in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        y = y.to(DEVICE)  # [B,1]

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            batch_loss = 0.0
            for xs, yi in zip(xs_list, y):
                xs = xs.to(DEVICE, non_blocking=True)
                yi = yi.view(1)

                logits_t = forward_logits(model, xs)
                target_t = yi.expand_as(logits_t)
                loss = loss_fn(logits_t, target_t)
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


def build_model_for_train(unfreeze_last_k: int = 1, train_classifier: bool = True) -> SwinForImageClassification:
    model = SwinForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    for p in model.parameters():
        p.requires_grad = False

    if train_classifier:
        for p in model.classifier.parameters():
            p.requires_grad = True

    unfreeze_last_k = int(unfreeze_last_k)
    if unfreeze_last_k < 0:
        unfreeze_last_k = 0
    if unfreeze_last_k > 4:
        unfreeze_last_k = 4

    stages_to_unfreeze = list(range(4 - unfreeze_last_k, 4))
    prefixes = [f"swin.encoder.layers.{s}." for s in stages_to_unfreeze]
    alt_prefixes = [f"swin.layers.{s}." for s in stages_to_unfreeze]
    prefixes_all = prefixes + alt_prefixes

    unfrozen_tensors = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in prefixes_all):
            p.requires_grad = True
            unfrozen_tensors += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Unfreeze] last_k={unfreeze_last_k} stages={stages_to_unfreeze} tensors_unfrozen={unfrozen_tensors}")
    print(f"Trainable params: {trainable:,} / {total:,}")
    if trainable > 60_000_000:
        print("⚠️ WARNING: trainable params가 너무 큽니다. unfreeze_last_k/prefix 확인 필요")

    return model


def _build_csv_from_folds(folds_csv: str, train_out="train_from_folds.csv", val_out="val_from_folds.csv"):
    folds = pd.read_csv(folds_csv)
    if not {"path", "label", "fold"}.issubset(folds.columns):
        raise ValueError("folds.csv에는 path,label,fold 컬럼이 필요함")

    tr = folds[folds["fold"] == "train"][["path", "label"]].reset_index(drop=True)
    va = folds[folds["fold"] == "val"][["path", "label"]].reset_index(drop=True)

    tr.to_csv(train_out, index=False)
    va.to_csv(val_out, index=False)
    print(f"[FOLDS] train={len(tr)} val={len(va)} -> {train_out}, {val_out}")
    return train_out, val_out


def train_main(args):
    set_seed(SEED)
    print("Device:", DEVICE)
    print("MODEL:", MODEL_ID, "IMG_SIZE:", IMG_SIZE)
    print("TRAIN_ROOT:", TRAIN_ROOT)
    print("TRAIN_NUM_FRAMES:", args.train_num_frames, "INFER_NUM_FRAMES:", args.infer_num_frames)
    print("AGG:", args.agg, "TOPK_RATIO:", args.topk_ratio)
    print("FACE_CROP:", args.face_crop)

    # ✅ CSV 준비: folds.csv 우선
    if args.use_folds:
        train_csv, val_csv = _build_csv_from_folds(args.folds_csv)
    else:
        if not Path("train.csv").exists() or not Path("val.csv").exists() or args.remake_csv:
            make_train_val_csv(TRAIN_ROOT, "train.csv", "val.csv", test_size=args.val_ratio, use_ratio=args.use_ratio)
        train_csv, val_csv = "train.csv", "val.csv"

    model = build_model_for_train(unfreeze_last_k=args.unfreeze_last_k)

    train_ds = MediaDataset(train_csv, TRAIN_ROOT, train_transform, num_frames=args.train_num_frames, face_crop=args.face_crop)
    val_ds   = MediaDataset(val_csv,   TRAIN_ROOT, val_transform,   num_frames=args.train_num_frames, face_crop=args.face_crop)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_media, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_media, pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    loss_fn = nn.BCEWithLogitsLoss()

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    start_epoch = 1
    best_auc = -1.0
    if args.resume:
        if CKPT_PATH.exists():
            last_ep, best_auc = load_ckpt(CKPT_PATH, model, optimizer, scheduler, scaler)
            start_epoch = last_ep + 1
            print(f"✅ resumed from {CKPT_PATH} (next epoch={start_epoch}, best_auc={best_auc:.4f})")
        else:
            print(f"⚠️ --resume 지정했지만 ckpt가 없음: {CKPT_PATH} (새로 시작)")

    # 학습
    for ep in range(start_epoch, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn)

        # ✅ 마지막 epoch에서 val 예측 저장(원하면 매 epoch 저장도 가능)
        save_pred = None
        if args.save_val_pred and (ep == args.epochs):
            save_pred = str(Path(args.save_val_pred).resolve())

        va_loss, va_auc = eval_auc(model, val_loader, loss_fn, args.agg, args.topk_ratio, save_pred_path=save_pred)

        print(f"[{ep}/{args.epochs}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_auc={va_auc:.4f}")

        torch.save(model.state_dict(), MODEL_OUT.parent / f"model_ep{ep}.pt")

        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), MODEL_OUT)
            print("✅ saved best ->", MODEL_OUT)

        save_ckpt(CKPT_PATH, model, optimizer, scheduler, scaler, ep, best_auc)

    if not MODEL_OUT.exists():
        torch.save(model.state_dict(), MODEL_OUT)
        print("✅ saved(last) ->", MODEL_OUT)

    print("Done. best_auc:", best_auc, "weight:", MODEL_OUT)
    print("Last ckpt:", CKPT_PATH)


# =========================
# 내부 infer (제출/디버깅 겸용)
# =========================
@torch.no_grad()
def predict_one_file(model, file_path: Path, num_frames: int, face_crop: bool, agg: str, topk_ratio: float) -> float:
    frames = read_rgb_frames(file_path, num_frames=num_frames)
    if len(frames) == 0:
        return 0.5

    proc = []
    for f in frames:
        if face_crop:
            f = crop_face_haar(f, margin=0.35)
        proc.append(infer_transform(f))
    xs = torch.stack(proc, dim=0).to(DEVICE, non_blocking=True)

    logits_t = forward_logits(model, xs)
    agg_logit = aggregate_logits(logits_t, agg, topk_ratio)
    prob_fake = torch.sigmoid(agg_logit).item()
    return float(max(0.0, min(1.0, prob_fake)))


def infer_main(args):
    print("Device:", DEVICE)
    print("MODEL:", MODEL_ID, "IMG_SIZE:", IMG_SIZE)
    print("Loading weights:", args.weights)
    print("INFER_NUM_FRAMES:", args.infer_num_frames)
    print("AGG:", args.agg, "TOPK_RATIO:", args.topk_ratio)
    print("FACE_CROP:", args.face_crop)

    model = SwinForImageClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True
    ).to(DEVICE)

    sd = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()

    test_root = Path(args.test_root)
    files = sorted([p for p in test_root.rglob("*") if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS)])
    print("files:", len(files))

    rows = []
    for fp in tqdm(files, desc="Infer"):
        prob = predict_one_file(
            model, fp,
            num_frames=args.infer_num_frames,
            face_crop=args.face_crop,
            agg=args.agg,
            topk_ratio=args.topk_ratio
        )
        rows.append({"filename": fp.name, "prob": prob})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("saved:", out_csv)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, default=EPOCHS)
    p_train.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p_train.add_argument("--lr", type=float, default=LR)
    p_train.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p_train.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--train_num_frames", type=int, default=TRAIN_NUM_FRAMES)
    p_train.add_argument("--infer_num_frames", type=int, default=INFER_NUM_FRAMES)
    p_train.add_argument("--agg", type=str, default=AGG)
    p_train.add_argument("--topk_ratio", type=float, default=TOPK_RATIO)
    p_train.add_argument("--unfreeze_last_k", type=int, default=1)
    p_train.add_argument("--val_ratio", type=float, default=0.2)
    p_train.add_argument("--remake_csv", action="store_true")
    p_train.add_argument("--face_crop", action="store_true", help="Haar 얼굴 크롭 사용")
    p_train.add_argument("--resume", action="store_true", help="ckpt.pt로부터 이어서 학습")
    p_train.add_argument("--use_ratio", type=float, default=0.5, help="기존 train.csv 만들 때만 사용 (folds 사용 시 무시)")

    # ✅ folds 사용 옵션
    p_train.add_argument("--use_folds", action="store_true", help="folds.csv 기반 split 사용")
    p_train.add_argument("--folds_csv", type=str, default="folds.csv", help="folds.csv 경로")

    # ✅ val 예측 저장(마지막 epoch에 pred_clean.csv 생성)
    p_train.add_argument("--save_val_pred", type=str, default="", help="예: pred_clean.csv (마지막 epoch val 예측 저장)")

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--weights", type=str, required=True)
    p_infer.add_argument("--test_root", type=str, default=str(TEST_ROOT))
    p_infer.add_argument("--out_csv", type=str, default=str(OUT_DIR / "submission.csv"))
    p_infer.add_argument("--infer_num_frames", type=int, default=INFER_NUM_FRAMES)
    p_infer.add_argument("--agg", type=str, default=AGG)
    p_infer.add_argument("--topk_ratio", type=float, default=TOPK_RATIO)
    p_infer.add_argument("--face_crop", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        # argparse 기본값 ""이면 None 취급
        if args.save_val_pred == "":
            args.save_val_pred = None
        train_main(args)
    elif args.cmd == "infer":
        infer_main(args)
