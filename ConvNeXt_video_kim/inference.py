"""
딥페이크 탐지 추론 코드

사용 예시 (PowerShell):
python inference.py --model_path .\checkpoints\model_1\best_model.pt --test_dir .\open\test_data --out_csv .\output\submission.csv --device cpu --agg topkmean --topk_ratio 0.25 --num_frames 8
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import DeepfakeDetector
from face_detector import FaceDetector


class TestDataset(torch.utils.data.Dataset):
    """테스트 데이터셋 - 추론용"""
    def __init__(self, test_dir, transform=None, use_face_detection=True, num_frames=8, image_size=224):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.num_frames = num_frames
        self.image_size = image_size

        self.face_detector = FaceDetector() if use_face_detection else None
        self.files = sorted(list(self.test_dir.glob("*")))

        # 비디오 확장자
        self.video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = file_path.name

        if file_path.suffix.lower() in self.video_exts:
            frames = self._load_video(file_path)    # [N,C,H,W]
        else:
            frames = self._load_image(file_path)    # [1,C,H,W]

        return frames, file_name

    def _load_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            return torch.zeros(1, 3, self.image_size, self.image_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.use_face_detection and self.face_detector:
            img = self.face_detector.crop_face_with_fallback(
                img, target_size=(self.image_size, self.image_size)
            )
        else:
            h, w = img.shape[:2]
            size = min(h, w)
            y1 = (h - size) // 2
            x1 = (w - size) // 2
            img = img[y1:y1+size, x1:x1+size]
            img = cv2.resize(img, (self.image_size, self.image_size))

        if not isinstance(img, np.ndarray) or len(img.shape) != 3 or img.shape[0] == 0:
            return torch.zeros(1, 3, self.image_size, self.image_size)

        if self.transform:
            img = self.transform(image=img)['image']
        else:
            # 안전장치(보통 transform 항상 있음)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img.unsqueeze(0)  # [1,C,H,W]

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return torch.zeros(1, 3, self.image_size, self.image_size)

        # 균등 샘플링
        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ✅ 학습과 동일하게 fallback 포함 크롭
            if self.use_face_detection and self.face_detector:
                frame = self.face_detector.crop_face_with_fallback(
                    frame, target_size=(self.image_size, self.image_size)
                )
            else:
                frame = cv2.resize(frame, (self.image_size, self.image_size))

            if frame is None or not isinstance(frame, np.ndarray) or frame.shape[0] == 0:
                continue

            if self.transform:
                frame = self.transform(image=frame)['image']
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return torch.zeros(1, 3, self.image_size, self.image_size)

        return torch.stack(frames)  # [N,C,H,W]


def aggregate_probs(probs_n1: torch.Tensor, agg: str = "median", topk_ratio: float = 0.25) -> float:
    """
    probs_n1: [N,1] 또는 [N]
    """
    p = probs_n1.view(-1)  # [N]

    if agg == "mean":
        return p.mean().item()
    if agg == "median":
        return p.median().item()
    if agg == "topkmean":
        k = max(1, int(p.numel() * topk_ratio))
        return torch.topk(p, k=k).values.mean().item()

    raise ValueError(f"Unknown agg: {agg}")


def inference(
    model_path: str,
    test_dir: str,
    output_csv: str,
    model_name: str = "convnext_small",
    image_size: int = 224,
    use_face_detection: bool = True,
    num_frames: int = 8,
    device: str = "cpu",
    agg: str = "median",
    topk_ratio: float = 0.25,
):
    device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    dataset = TestDataset(
        test_dir=test_dir,
        transform=transform,
        use_face_detection=use_face_detection,
        num_frames=num_frames,
        image_size=image_size
    )

    print(f"Total test files: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(f"테스트 폴더가 비었거나 경로가 틀림: {test_dir}")

    model = DeepfakeDetector(model_name=model_name, pretrained=False, num_classes=1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"Model loaded: {model_path}")
    print(f"Aggregation: {agg} (topk_ratio={topk_ratio})")

    results = []
    with torch.no_grad():
        for frames, file_name in tqdm(dataset, desc="Inference"):
            frames = frames.to(device)  # [N,C,H,W]
            logits = model(frames)      # [N,1]
            probs = torch.sigmoid(logits)

            prob = aggregate_probs(probs, agg=agg, topk_ratio=topk_ratio)

            results.append({"filename": file_name, "prob": prob})

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results, columns=["filename", "prob"])
    df.to_csv(out_path, index=False)

    print(f"✓ Saved: {out_path} (rows={len(df)})")
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--model_name", type=str, default="convnext_small")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--use_face_detection", action="store_true")
    p.add_argument("--no_face_detection", action="store_true")
    p.add_argument("--agg", type=str, default="median", choices=["mean", "median", "topkmean"])
    p.add_argument("--topk_ratio", type=float, default=0.25)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_face = True
    if args.no_face_detection:
        use_face = False
    elif args.use_face_detection:
        use_face = True

    inference(
        model_path=args.model_path,
        test_dir=args.test_dir,
        output_csv=args.out_csv,
        model_name=args.model_name,
        image_size=args.image_size,
        use_face_detection=use_face,
        num_frames=args.num_frames,
        device=args.device,
        agg=args.agg,
        topk_ratio=args.topk_ratio,
    )
