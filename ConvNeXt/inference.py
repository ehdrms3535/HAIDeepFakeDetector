"""
딥페이크 탐지 추론
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


class TestDataset(Dataset):
    """테스트 데이터셋"""
    def __init__(self, test_dir, transform=None, use_face_detection=True, num_frames=8, image_size=224):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.num_frames = num_frames
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
        
        # 얼굴 검출 (학습과 동일한 방식)
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
        """비디오에서 프레임 추출"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 균등하게 프레임 선택
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
                face = self.face_detector.detect_face(frame)
                if face is not None:
                    frame = face
            
            # 리사이즈 (frame이 유효한지 확인)
            if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0:
                frame = cv2.resize(frame, (self.image_size, self.image_size))
            else:
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


def inference(model_path, test_dir, output_csv, model_name="convnext_small", 
              image_size=224, batch_size=32, use_face_detection=True, 
              num_frames=8, device="cuda"):
    """추론 실행"""
    
    # Device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transform
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Dataset
    dataset = TestDataset(
        test_dir=test_dir,
        transform=transform,
        use_face_detection=use_face_detection,
        num_frames=num_frames,
        image_size=image_size
    )
    
    print(f"\nTotal test files: {len(dataset)}")
    
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
            
            # 비디오 추론 개선: 중앙값 사용 (평균보다 아웃라이어에 강함)
            if probs.shape[0] > 1:  # 비디오 (여러 프레임)
                # 중앙값 사용
                avg_prob = probs.median().item()
            else:  # 이미지 (단일 프레임)
                avg_prob = probs.mean().item()
            
            results.append({
                'filename': file_name,
                'probability': avg_prob
            })
    
    # CSV 저장
    df = pd.DataFrame(results)
    df.columns = ['filename', 'prob']  # 제출 양식에 맞게 컬럼명 변경
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Results saved: {output_csv}")
    print(f"Total predictions: {len(df)}")
    
    return df
