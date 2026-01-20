# train.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from transformers import SwinForImageClassification, get_cosine_schedule_with_warmup


# =========================
# 기본 설정 (필요시 argparse로 오버라이드)
# =========================
SEED = 42

MODEL_ID = "microsoft/swin-base-patch4-window7-224"
IMG_SIZE = 224

# 테스트 비디오가 5초 이하라고 했으니, 너무 많은 프레임은 중복/비용만 증가할 수 있음
TRAIN_NUM_FRAMES = 8
INFER_NUM_FRAMES = 16

# 집계(후처리 허용)
AGG = "topkmean"      # mean | max | topkmean
TOPK_RATIO = 0.15     # topkmean 상위 비율(0.1~0.2 추천)

EPOCHS = 10
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0

# 경로
TRAIN_ROOT = Path("./train_data")
TEST_ROOT  = Path("./test_data")
MODEL_OUT  = Path("./model/model.pt")
CKPT_PATH  = Path("./model/ckpt.pt")   # ✅ resume용 체크포인트
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
# ✅ 체크포인트 저장/로드 (resume)
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

    # RNG 복구 (가능한 한 재현)
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
    """
    rgb: HxWx3 (RGB)
    return: cropped RGB (square-ish, padded if needed)
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return rgb  # fallback: 원본

    # 가장 큰 얼굴 선택
    x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])

    # margin 확장
    cx = x + bw / 2
    cy = y + bh / 2
    side = max(bw, bh) * (1.0 + 2.0 * margin)

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))

    # 패딩 포함 안전 크롭
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
    """
    - 이미지: 1장 반환
    - 비디오: 결정적 균등 샘플로 num_frames장
    """
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTS:
        from PIL import Image
        try:
            img = Image.open(file_path).convert("RGB")
            return [np.array(img)]
        except Exception:
            return []

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: List[np.ndarray] = []

        # frame_count가 0인 깨진 파일: 순차 읽고 균등 샘플 대체(결정적)
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

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
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

infer_transform = val_transform  # ✅ 추론은 단 1회(=TTA 금지)로 고정


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
# CSV 생성: fake=1, real=0
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

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df["label"])
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    print(f"[CSV] train={len(train_df)} val={len(val_df)} saved -> {out_train}, {out_val}")


# =========================
# Dataset (파일 1개 -> [T,3,H,W], label)
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

        return xs, torch.tensor([y], dtype=torch.float32)


def collate_media(batch):
    xs_list, ys_list = zip(*batch)
    return list(xs_list), torch.cat(ys_list, dim=0)


def forward_logits(model, xs: torch.Tensor) -> torch.Tensor:
    """
    xs: [T,3,H,W] -> logits: [T]
    """
    outputs = model(pixel_values=xs)
    logits = outputs.logits
    logits = logits.squeeze()
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 1:
        logits = logits.view(-1)
    return logits


@torch.no_grad()
def eval_auc(model, loader, loss_fn, agg_method: str, topk_ratio: float):
    model.eval()
    ys, ps, losses = [], [], []

    for xs_list, y in loader:
        y = y.to(DEVICE)  # [B,1]
        for xs, yi in zip(xs_list, y):
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

    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return float(np.mean(losses)), float(auc)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn):
    model.train()
    losses = []

    for xs_list, y in tqdm(loader, desc="Train", leave=False):
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


def build_model_for_train(unfreeze_last_k_stages: int = 2) -> SwinForImageClassification:
    model = SwinForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # classifier always train
    for p in model.classifier.parameters():
        p.requires_grad = True

    # unfreeze last k stages
    if hasattr(model, "swin"):
        layers = None
        if hasattr(model.swin, "encoder") and hasattr(model.swin.encoder, "layers"):
            layers = model.swin.encoder.layers
        elif hasattr(model.swin, "layers"):
            layers = model.swin.layers

        if layers is not None:
            k = max(0, min(unfreeze_last_k_stages, len(layers)))
            for i in range(len(layers) - k, len(layers)):
                for p in layers[i].parameters():
                    p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
    return model


def train_main(args):
    set_seed(SEED)
    print("Device:", DEVICE)
    print("MODEL:", MODEL_ID, "IMG_SIZE:", IMG_SIZE)
    print("TRAIN_NUM_FRAMES:", args.train_num_frames, "INFER_NUM_FRAMES:", args.infer_num_frames)
    print("AGG:", args.agg, "TOPK_RATIO:", args.topk_ratio)
    print("FACE_CROP:", args.face_crop)

    if not Path("train.csv").exists() or not Path("val.csv").exists() or args.remake_csv:
        make_train_val_csv(TRAIN_ROOT, "train.csv", "val.csv", test_size=args.val_ratio)

    model = build_model_for_train(unfreeze_last_k_stages=args.unfreeze_last_k)

    train_ds = MediaDataset("train.csv", TRAIN_ROOT, train_transform, num_frames=args.train_num_frames, face_crop=args.face_crop)
    val_ds   = MediaDataset("val.csv",   TRAIN_ROOT, val_transform,   num_frames=args.train_num_frames, face_crop=args.face_crop)

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

    # ✅ resume
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
        va_loss, va_auc = eval_auc(model, val_loader, loss_fn, args.agg, args.topk_ratio)

        print(f"[{ep}/{args.epochs}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_auc={va_auc:.4f}")

        # ✅ 매 epoch 끝나면 ckpt 저장 (중단 대비)
        save_ckpt(CKPT_PATH, model, optimizer, scheduler, scaler, ep, best_auc)

        # ✅ best 모델 저장
        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), MODEL_OUT)
            print("✅ saved best ->", MODEL_OUT)

    # 혹시 model.pt가 없으면 last 저장
    if not MODEL_OUT.exists():
        torch.save(model.state_dict(), MODEL_OUT)
        print("✅ saved(last) ->", MODEL_OUT)

    print("Done. best_auc:", best_auc, "weight:", MODEL_OUT)
    print("Last ckpt:", CKPT_PATH)


# =========================
# 내부 infer (디버깅용) - 대회 제출은 inference.py 사용 권장
# =========================
@torch.no_grad()
def predict_one_file(model, file_path: Path, num_frames: int, face_crop: bool, agg: str, topk_ratio: float) -> float:
    frames = read_rgb_frames(file_path, num_frames=num_frames)
    if len(frames) == 0:
        return 0.5  # ✅ 깨진 영상/프레임 실패 안전값

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
    print("test files:", len(files))

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
    p_train.add_argument("--unfreeze_last_k", type=int, default=2)
    p_train.add_argument("--val_ratio", type=float, default=0.2)
    p_train.add_argument("--remake_csv", action="store_true")
    p_train.add_argument("--face_crop", action="store_true", help="Haar 얼굴 크롭 사용")
    p_train.add_argument("--resume", action="store_true", help="ckpt.pt로부터 이어서 학습")

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
        train_main(args)
    elif args.cmd == "infer":
        infer_main(args)
