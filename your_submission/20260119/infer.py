import argparse
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import SwinForImageClassification
from PIL import Image, ImageFilter

# ===== 너 학습 설정에 맞춰 동일하게 =====
MODEL_ID = "microsoft/swin-base-patch4-window7-224"
IMG_SIZE = 224

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

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
        rgb = cv2.copyMakeBorder(rgb, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x1 += pad_l; x2 += pad_l
        y1 += pad_t; y2 += pad_t

    return rgb[y1:y2, x1:x2]

def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)

def read_rgb_frames(file_path: Path, num_frames: int):
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
        frames = []

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
            idxs = uniform_frame_indices(len(tmp), num_frames)
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

def forward_logits(model, xs: torch.Tensor) -> torch.Tensor:
    # xs: [T,3,H,W]
    out = model(pixel_values=xs)
    logits = out.logits.squeeze()
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 1:
        logits = logits.view(-1)
    return logits  # [T]

def aggregate_logits(logits: torch.Tensor, method="topkmean", topk_ratio=0.15) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    m = method.lower()
    if m == "mean":
        return logits.mean()
    if m == "max":
        return logits.max()
    if m in ("topkmean", "topk"):
        k = max(1, int(np.ceil(float(topk_ratio) * logits.numel())))
        return torch.topk(logits, k=k).values.mean()
    raise ValueError(f"Unknown agg method: {method}")

@torch.no_grad()
def predict_one(model, fp: Path, num_frames: int, face_crop: bool, agg: str, topk_ratio: float) -> float:
    frames = read_rgb_frames(fp, num_frames=num_frames)
    if len(frames) == 0:
        return 0.5

    proc = []
    for f in frames:
        if face_crop:
            f = crop_face_haar(f, margin=0.35)
        proc.append(val_transform(f))
    xs = torch.stack(proc, dim=0).to(DEVICE, non_blocking=True)

    logits_t = forward_logits(model, xs)
    agg_logit = aggregate_logits(logits_t, agg, topk_ratio)
    return float(torch.sigmoid(agg_logit).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds_csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_csv", default="pred_clean.csv")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--agg", type=str, default="topkmean")
    ap.add_argument("--topk_ratio", type=float, default=0.15)
    ap.add_argument("--face_crop", action="store_true")
    ap.add_argument("--which_fold", type=str, default="val")  # val만
    args = ap.parse_args()

    folds = pd.read_csv(args.folds_csv)
    if "path" not in folds.columns or "fold" not in folds.columns:
        raise RuntimeError("folds_csv에는 최소 path, fold 컬럼이 있어야 함")

    v = folds[folds["fold"].astype(str).str.lower() == args.which_fold.lower()].copy()
    print(f"rows fold={args.which_fold}:", len(v))

    model = SwinForImageClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True
    ).to(DEVICE)

    sd = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()

    root = Path(args.data_root)

    rows = []
    for rel in tqdm(v["path"].astype(str).tolist(), desc=f"Infer fold={args.which_fold}"):
        fp = root / rel
        p = predict_one(model, fp, args.num_frames, args.face_crop, args.agg, args.topk_ratio)
        rows.append({"path": rel.replace("\\", "/"), "filename": Path(rel).name, "pred": p})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)

if __name__ == "__main__":
    main()
