# infer_facecrop_haar.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification
from PIL import Image

# (안정화) torch/cv2 스레드 제한
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ===== Face crop (OpenCV Haar) =====
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_face_haar(
    img: Image.Image,
    margin: float = 0.35,
    make_square: bool = True,
    fill_color=(0, 0, 0)
) -> Image.Image:
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

# ===== 설정 (train_facecrop_haar.py랑 맞추기) =====
MODEL_ID = "google/vit-base-patch16-224-in21k"
IMG_SIZE = 224
NUM_FRAMES = 16            # 학습 NUM_FRAMES와 동일하게
AGG = "topkmean"
TOPK = 2

TEST_ROOT = Path("./test_data")
MODEL_PATH = Path("./model/model_facecrop.pt")
OUT_CSV = Path("./output/baseline_submission_facecrop.csv")
OUT_CSV.parent.mkdir(exist_ok=True, parents=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_rgb_frames(file_path: Path, num_frames: int = NUM_FRAMES) -> list[np.ndarray]:
    ext = file_path.suffix.lower()

    if ext in IMAGE_EXTS:
        img = Image.open(file_path).convert("RGB")
        img = crop_face_haar(img)
        return [np.array(img)]

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return []

            # ✅ 2초당 1프레임
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 1e-6:
                fps = 30.0
            step = max(int(round(fps * 2.0)), 1)
            idxs = np.arange(0, total, step, dtype=int)

            if num_frames is not None and len(idxs) > num_frames:
                idxs = idxs[:num_frames]

            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil = crop_face_haar(pil)
                frames.append(np.array(pil))
            return frames
        finally:
            cap.release()

    return []

# ✅ val/infer는 학습 val_transform과 동일하게
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    raise ValueError(method)

@torch.no_grad()
def predict_file(model, fp: Path) -> float:
    frames = read_rgb_frames(fp, num_frames=NUM_FRAMES)
    if len(frames) == 0:
        return 0.0

    xs = torch.stack([transform(f) for f in frames], dim=0).to(DEVICE)  # [T,3,224,224]
    outputs = model(xs)

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    logits = logits.squeeze()
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 1:
        logits = logits.view(-1)

    logit = aggregate_logits(logits, AGG, TOPK)
    prob = torch.sigmoid(logit).item()
    return float(max(0.0, min(1.0, prob)))

def main():
    print("Device:", DEVICE)
    print("Loading ViT:", MODEL_ID)
    print("Loading weights:", MODEL_PATH)

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True
    ).to(DEVICE)

    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    files = sorted([
        p for p in TEST_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS)
    ])
    print("test files:", len(files))

    rows = []
    for fp in tqdm(files, desc="Infer"):
        prob = predict_file(model, fp)
        rows.append({"filename": fp.name, "prob": prob})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("saved:", OUT_CSV)
    print(df.head())

if __name__ == "__main__":
    main()
