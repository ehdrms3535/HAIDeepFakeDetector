import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import SwinForImageClassification

# -------- 기본 설정 --------
MODEL_ID = "microsoft/swin-base-patch4-window7-224"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="val_hard 폴더")
    ap.add_argument("--weights", type=str, required=True, help="model/model.pt")
    ap.add_argument("--out_csv", type=str, default="pred_hard.csv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = args.device
    root = Path(args.root)

    files = list_images(root)
    print("files:", len(files), "from", root.resolve())

    model = SwinForImageClassification.from_pretrained(
        MODEL_ID, num_labels=1, ignore_mismatched_sizes=True
    ).to(device)

    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    rows = []
    bs = args.batch_size

    for i in tqdm(range(0, len(files), bs), desc="Infer"):
        batch_files = files[i:i+bs]

        xs = []
        fnames = []
        for fp in batch_files:
            try:
                img = Image.open(fp).convert("RGB")
                x = val_transform(img)
                xs.append(x)
                fnames.append(fp.name)
            except Exception:
                # 실패 시 안전값
                fnames.append(fp.name)
                xs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

        x = torch.stack(xs, dim=0).to(device, non_blocking=True)  # [B,3,H,W]

        logits = model(pixel_values=x).logits.squeeze(-1)  # [B]
        prob = torch.sigmoid(logits).detach().cpu().numpy()

        for f, p in zip(fnames, prob):
            rows.append({"filename": f, "pred": float(p)})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("saved ->", args.out_csv)

if __name__ == "__main__":
    main()
