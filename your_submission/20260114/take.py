import random
from pathlib import Path

import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from transformers import SwinForImageClassification

# ======================
# 설정 (너 환경에 맞게만 수정)
# ======================
SEED = 42

MODEL_ID = "microsoft/swin-large-patch4-window12-384"
IMG_SIZE = 384
NUM_FRAMES = 5

REAL_DIR = Path("../train_data/real")   # 너 구조에 맞게 수정
FAKE_DIR = Path("../train_data/fake")   # 너 구조에 맞게 수정

MODEL_PATH = Path("./model/model.pt")   # 가중치 경로

K = 30  # real/fake 각각 샘플링 개수

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 전처리
# ======================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def uniform_frame_indices(total_frames: int, num_frames: int):
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

def read_rgb_frames(file_path: Path, num_frames=NUM_FRAMES):
    ext = file_path.suffix.lower()

    # image
    if ext in IMAGE_EXTS:
        from PIL import Image
        try:
            img = Image.open(file_path).convert("RGB")
            return [np.array(img)]
        except Exception:
            return []

    # video
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # fallback: frame_count가 0이면 끝까지 읽고 균등샘플
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
            return [cv2.cvtColor(tmp[int(i)], cv2.COLOR_BGR2RGB) for i in idxs]

        idxs = uniform_frame_indices(total, num_frames)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    return []

@torch.no_grad()
def predict_file(model, fp: Path, num_frames=NUM_FRAMES) -> float:
    frames = read_rgb_frames(fp, num_frames)
    if len(frames) == 0:
        return 0.0

    xs = torch.stack([transform(f) for f in frames], dim=0).to(DEVICE)
    logits = model(xs).logits.squeeze()

    # logits: [T] 또는 scalar
    if logits.dim() == 0:
        logit = logits
    else:
        logit = logits.mean()

    # ✅ 이 값은 "학습을 fake=1로 했으면" P(fake)
    return float(torch.sigmoid(logit).item())

def pick_files(root: Path, k: int):
    exts = IMAGE_EXTS | VIDEO_EXTS
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if len(files) == 0:
        return []
    random.Random(SEED).shuffle(files)
    return files[: min(k, len(files))]

def main():
    print("Device:", DEVICE)
    print("MODEL_ID:", MODEL_ID)
    print("MODEL_PATH:", MODEL_PATH.resolve())

    # 1) 파일 존재 체크
    if not MODEL_PATH.exists():
        print("[ERR] model.pt not found:", MODEL_PATH)
        return
    if not REAL_DIR.exists():
        print("[ERR] REAL_DIR not found:", REAL_DIR)
        return
    if not FAKE_DIR.exists():
        print("[ERR] FAKE_DIR not found:", FAKE_DIR)
        return

    # 2) 모델 로드
    model = SwinForImageClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True
    ).to(DEVICE)

    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)  # 일단 false로(헤드 mismatch 방지)
    model.eval()

    # 3) 샘플 파일 뽑기
    real_files = pick_files(REAL_DIR, K)
    fake_files = pick_files(FAKE_DIR, K)
    print(f"Picked files: real={len(real_files)} fake={len(fake_files)} (K={K})")

    if len(real_files) == 0 or len(fake_files) == 0:
        print("[ERR] 샘플링 실패. 폴더 경로나 확장자 확인.")
        return

    # 4) 예측
    real_probs = [predict_file(model, p) for p in tqdm(real_files, desc="REAL")]
    fake_probs = [predict_file(model, p) for p in tqdm(fake_files, desc="FAKE")]

    print("\n===== RESULT =====")
    print("REAL mean:", float(np.mean(real_probs)), "min:", float(np.min(real_probs)), "max:", float(np.max(real_probs)))
    print("FAKE mean:", float(np.mean(fake_probs)), "min:", float(np.min(fake_probs)), "max:", float(np.max(fake_probs)))

    print("\n[샘플 5개]")
    for p, pr in list(zip(real_files, real_probs))[:5]:
        print("REAL", p.name, pr)
    for p, pr in list(zip(fake_files, fake_probs))[:5]:
        print("FAKE", p.name, pr)

if __name__ == "__main__":
    main()
