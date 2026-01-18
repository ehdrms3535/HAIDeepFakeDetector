"""
향상된 딥페이크 탐지 추론 - 비디오 정확도 개선
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import DeepfakeDetector
from face_detector import FaceDetector


class EnhancedTestDataset(Dataset):
    """향상된 테스트 데이터셋 - 비디오 프레임 증가"""
    def __init__(self, test_dir, transform=None, use_face_detection=True, num_frames=16, image_size=224):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.num_frames = num_frames  # 더 많은 프레임
        self.image_size = image_size
        
        if use_face_detection:
            self.face_detector = FaceDetector()
        
        # 모든 테스트 파일
        self.files = sorted(list(self.test_dir.glob("*")))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = file_path.name
        
        # 비디오 확장자
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        
        if file_path.suffix.lower() in video_exts:
            frames = self._load_video(file_path)
        else:
            frames = self._load_image(file_path)
        
        return frames, file_name
    
    def _load_image(self, image_path):
        """이미지 로드"""
        img = cv2.imread(str(image_path))
        if img is None:
            return torch.zeros(1, 3, self.image_size, self.image_size)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 얼굴 검출
        if self.use_face_detection:
            img = self.face_detector.crop_face_with_fallback(
                img,
                target_size=(self.image_size, self.image_size)
            )
        else:
            # 중앙 크롭 후 리사이즈
            h, w = img.shape[:2]
            size = min(h, w)
            y1 = (h - size) // 2
            x1 = (w - size) // 2
            img = img[y1:y1+size, x1:x1+size]
            img = cv2.resize(img, (self.image_size, self.image_size))
        
        # 유효성 확인
        if not isinstance(img, np.ndarray) or len(img.shape) != 3 or img.shape[0] == 0:
            return torch.zeros(1, 3, self.image_size, self.image_size)
        
        # Transform
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img.unsqueeze(0)  # [1, C, H, W]
    
    def _load_video(self, video_path):
        """비디오에서 프레임 추출 - 더 많은 프레임 사용"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 더 많은 프레임 추출
        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 얼굴 검출
            if self.use_face_detection:
                frame = self.face_detector.crop_face_with_fallback(
                    frame,
                    target_size=(self.image_size, self.image_size)
                )
            else:
                # 중앙 크롭 후 리사이즈
                h, w = frame.shape[:2]
                size = min(h, w)
                y1 = (h - size) // 2
                x1 = (w - size) // 2
                frame = frame[y1:y1+size, x1:x1+size]
                frame = cv2.resize(frame, (self.image_size, self.image_size))
            
            # 유효성 확인
            if not isinstance(frame, np.ndarray) or len(frame.shape) != 3 or frame.shape[0] == 0:
                continue
            
            # Transform
            if self.transform:
                frame = self.transform(image=frame)['image']
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            # 빈 프레임 반환
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            return dummy
        
        # [N, C, H, W]
        return torch.stack(frames)


def inference_enhanced(model_path, test_dir, output_csv, model_name="convnext_small", 
                      image_size=224, batch_size=32, use_face_detection=True, 
                      num_frames=16, device="cuda"):
    """향상된 추론 실행 - 비디오 정확도 개선"""
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transform
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Dataset
    dataset = EnhancedTestDataset(
        test_dir=test_dir,
        transform=transform,
        use_face_detection=use_face_detection,
        num_frames=num_frames,
        image_size=image_size
    )
    
    print(f"\nTotal test files: {len(dataset)}")
    print(f"Frames per video: {num_frames}")
    
    # Model
    model = DeepfakeDetector(model_name=model_name, pretrained=False, num_classes=1)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model_path}")
    
    # Inference
    results = []
    
    with torch.no_grad():
        for frames, file_name in tqdm(dataset, desc="Inference"):
            frames = frames.to(device)  # [N, C, H, W]
            
            # 배치로 처리
            logits = model(frames)  # [N, 1]
            probs = torch.sigmoid(logits)  # [N, 1]
            
            # 비디오 추론 개선: 여러 통계 사용
            if probs.shape[0] > 1:  # 비디오 (여러 프레임)
                # 평균, 중앙값, 최대값의 가중 평균
                mean_prob = probs.mean().item()
                median_prob = probs.median().item()
                max_prob = probs.max().item()
                
                # 가중 평균: 평균 50%, 중앙값 30%, 최대값 20%
                avg_prob = 0.5 * mean_prob + 0.3 * median_prob + 0.2 * max_prob
            else:  # 이미지 (단일 프레임)
                avg_prob = probs.mean().item()
            
            results.append({
                'filename': file_name,
                'probability': avg_prob
            })
    
    # CSV 저장
    df = pd.DataFrame(results)
    df.columns = ['filename', 'prob']
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Results saved: {output_csv}")
    print(f"Total predictions: {len(df)}")
    
    return df
