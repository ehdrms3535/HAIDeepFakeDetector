from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import SwinForImageClassification


# ====== 설정 ======
MODEL_ID = "microsoft/swin-tiny-patch4-window7-224"
IMG_SIZE = 224
NUM_FRAMES = 5

AGG = "topkmean"   # mean|max|topkmean
TOPK = 2

TEST_ROOT = Path("./test_data")
MODEL_PATH = Path("./model/model.pt")
OUT_CSV = Path("./output/baseline_submission.csv")
OUT_CSV.parent.mkdir(exist_ok=True, parents=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def uniform_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    if total_frames <= num_frames:
        return np.arange(total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


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


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
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
    print("Loading model:", MODEL_ID)

    model = SwinForImageClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True
    ).to(DEVICE)

    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    files = sorted([p for p in TEST_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS)])
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
