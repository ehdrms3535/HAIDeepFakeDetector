# inference.py
# 사용법:
#   python inference.py --weights ./model/model.pt --test_root ../test_data --out_csv ./output/submission.csv
#
# 옵션 예시:
#   python inference.py --weights ./model/model.pt --test_root ../test_data --out_csv ./output/submission.csv --num_frames 24 --agg topkmean --topk_ratio 0.2 --tta

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import SwinForImageClassification

# =========================
# 기본 설정
# =========================
SEED = 42
MODEL_ID_DEFAULT = "microsoft/swin-base-patch4-window7-224"
IMG_SIZE_DEFAULT = 224

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 프레임 인덱스 유틸
# =========================
def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


# =========================
# 비디오/이미지 읽기 (frame_count=0 fallback 포함)
# =========================
def read_rgb_frames(file_path: Path, num_frames: int, seed: int = SEED) -> List[np.ndarray]:
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

        # fallback: frame_count가 0/깨진 경우 -> reservoir sampling
        if total <= 0:
            rng = random.Random(seed + (hash(str(file_path)) % 10_000_000))
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
            return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in picked]

        # 정상: 균등 샘플
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
# Transform (안정적인 center-crop)
# =========================
def build_infer_transform(img_size: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =========================
# 집계 (로짓 기준)
# =========================
def aggregate_logits(logits: torch.Tensor, method: str = "topkmean", topk_ratio: float = 0.2) -> torch.Tensor:
    # logits: [T]
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


@torch.no_grad()
def forward_logits(model, xs: torch.Tensor) -> torch.Tensor:
    """
    xs: [T,3,H,W] -> logits: [T]
    """
    out = model(pixel_values=xs)
    logits = out.logits if hasattr(out, "logits") else out
    logits = logits.squeeze()

    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 1:
        logits = logits.view(-1)
    return logits


@torch.no_grad()
def predict_one_file(
    model,
    file_path: Path,
    num_frames: int,
    infer_transform,
    agg: str,
    topk_ratio: float,
    tta: bool,
) -> float:
    frames = read_rgb_frames(file_path, num_frames=num_frames, seed=SEED)
    if len(frames) == 0:
        return 0.0

    # 기본(원본)
    xs0 = torch.stack([infer_transform(f) for f in frames], dim=0).to(DEVICE, non_blocking=True)  # [T,3,H,W]
    logits0 = forward_logits(model, xs0)  # [T]
    agg0 = aggregate_logits(logits0, agg, topk_ratio)

    if not tta:
        p = torch.sigmoid(agg0).item()
        return float(max(0.0, min(1.0, p)))

    # TTA: 좌우 플립 1회 추가 평균
    xs1 = torch.flip(xs0, dims=[3])  # W flip
    logits1 = forward_logits(model, xs1)
    agg1 = aggregate_logits(logits1, agg, topk_ratio)

    # 로짓 평균(안정적)
    agg_logit = (agg0 + agg1) / 2.0
    p = torch.sigmoid(agg_logit).item()
    return float(max(0.0, min(1.0, p)))


def load_model(model_id: str, weights: str):
    model = SwinForImageClassification.from_pretrained(
        model_id,
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    sd = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="학습된 state_dict 경로 (예: ./model/model.pt)")
    parser.add_argument("--test_root", type=str, required=True, help="테스트 데이터 폴더")
    parser.add_argument("--out_csv", type=str, required=True, help="출력 CSV 경로 (filename,prob)")

    parser.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE_DEFAULT)

    parser.add_argument("--num_frames", type=int, default=24, help="비디오에서 뽑을 프레임 수(권장 16~32)")
    parser.add_argument("--agg", type=str, default="topkmean", choices=["mean", "max", "topkmean"])
    parser.add_argument("--topk_ratio", type=float, default=0.2, help="topkmean일 때 상위 비율(0.1~0.3 추천)")
    parser.add_argument("--tta", action="store_true", help="flip TTA 적용(느려지지만 점수 안정화에 도움)")

    args = parser.parse_args()

    set_seed(SEED)
    print("Device:", DEVICE)
    print("MODEL_ID:", args.model_id, "IMG_SIZE:", args.img_size)
    print("weights:", args.weights)
    print("num_frames:", args.num_frames, "agg:", args.agg, "topk_ratio:", args.topk_ratio, "tta:", args.tta)

    model = load_model(args.model_id, args.weights)
    infer_transform = build_infer_transform(args.img_size)

    test_root = Path(args.test_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in test_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS)])
    print("test files:", len(files))

    rows = []
    for fp in tqdm(files, desc="Infer"):
        prob = predict_one_file(
            model=model,
            file_path=fp,
            num_frames=args.num_frames,
            infer_transform=infer_transform,
            agg=args.agg,
            topk_ratio=args.topk_ratio,
            tta=args.tta,
        )
        rows.append({"filename": fp.name, "prob": prob})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)
    print(df.head())


if __name__ == "__main__":
    main()
