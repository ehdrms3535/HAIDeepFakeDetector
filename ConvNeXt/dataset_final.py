"""
최종 데이터셋 - 모든 데이터 통합
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import random

from face_detector import FaceDetector


class FinalDeepfakeDataset(Dataset):
    """모든 데이터를 통합한 최종 데이터셋"""
    
    def __init__(
        self,
        image_real_dir: str,
        image_fake_dir: str,
        video_real_dir: str,
        video_fake_dir: str,
        transform=None,
        use_face_detection: bool = True,
        num_frames_per_video: int = 16,
        image_size: int = 224,
        max_samples_per_class: int = None,
        sample_offset: int = 0  # 샘플 시작 위치
    ):
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.num_frames_per_video = num_frames_per_video
        self.image_size = image_size
        self.sample_offset = sample_offset  # offset 저장
        
        # 얼굴 검출기
        if use_face_detection:
            self.face_detector = FaceDetector()
        else:
            self.face_detector = None
        
        # 샘플 리스트
        self.samples = []
        
        # 이미지 확장자
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.jfif']
        video_exts = ['.mp4', '.avi', '.mov']
        
        print("Loading dataset...")
        
        # 1. 이미지 - REAL
        real_count = 0
        image_real_path = Path(image_real_dir)
        if image_real_path.exists():
            real_files = []
            for ext in image_exts:
                real_files.extend(list(image_real_path.glob(f"*{ext}")))
                real_files.extend(list(image_real_path.glob(f"*{ext.upper()}")))
            
            # 샘플링 (offset 적용)
            if max_samples_per_class:
                # 정렬해서 일관성 유지
                real_files = sorted(real_files)
                start_idx = sample_offset
                end_idx = start_idx + max_samples_per_class
                real_files = real_files[start_idx:end_idx]
            
            for file_path in real_files:
                self.samples.append((str(file_path), 0, 'image'))
                real_count += 1
        
        print(f"  Image REAL: {real_count}")
        
        # 2. 이미지 - FAKE
        fake_count = 0
        image_fake_path = Path(image_fake_dir)
        if image_fake_path.exists():
            fake_files = []
            for ext in image_exts:
                fake_files.extend(list(image_fake_path.glob(f"*{ext}")))
                fake_files.extend(list(image_fake_path.glob(f"*{ext.upper()}")))
            
            # 샘플링 (offset 적용)
            if max_samples_per_class:
                # 정렬해서 일관성 유지
                fake_files = sorted(fake_files)
                start_idx = sample_offset
                end_idx = start_idx + max_samples_per_class
                fake_files = fake_files[start_idx:end_idx]
            
            for file_path in fake_files:
                self.samples.append((str(file_path), 1, 'image'))
                fake_count += 1
        
        print(f"  Image FAKE: {fake_count}")
        
        # 3. 비디오 - REAL
        video_real_count = 0
        video_real_path = Path(video_real_dir)
        if video_real_path.exists():
            for ext in video_exts:
                for file_path in video_real_path.glob(f"*{ext}"):
                    self.samples.append((str(file_path), 0, 'video'))
                    video_real_count += 1
        
        print(f"  Video REAL: {video_real_count}")
        
        # 4. 비디오 - FAKE
        video_fake_count = 0
        video_fake_path = Path(video_fake_dir)
        if video_fake_path.exists():
            for ext in video_exts:
                for file_path in video_fake_path.glob(f"*{ext}"):
                    self.samples.append((str(file_path), 1, 'video'))
                    video_fake_count += 1
        
        print(f"  Video FAKE: {video_fake_count}")
        
        print(f"\nTotal samples: {len(self.samples)}")
        print(f"  REAL: {real_count + video_real_count}")
        print(f"  FAKE: {fake_count + video_fake_count}")
        
        # 셔플
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, path: str) -> np.ndarray:
        """이미지 로드"""
        try:
            image = Image.open(path).convert('RGB')
            return np.array(image)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def extract_frames_from_video(self, path: str) -> List[np.ndarray]:
        """비디오에서 프레임 추출"""
        cap = cv2.VideoCapture(path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []
        
        # 균등 샘플링
        indices = np.linspace(0, total_frames - 1, self.num_frames_per_video, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리 (얼굴 크롭)"""
        if self.use_face_detection and self.face_detector:
            image = self.face_detector.crop_face_with_fallback(
                image,
                target_size=(self.image_size, self.image_size)
            )
        else:
            # 중앙 크롭
            h, w = image.shape[:2]
            size = min(h, w)
            y1 = (h - size) // 2
            x1 = (w - size) // 2
            image = image[y1:y1+size, x1:x1+size]
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        return image
    
    def __getitem__(self, idx):
        file_path, label, file_type = self.samples[idx]
        
        try:
            if file_type == 'video':
                # 비디오: 프레임 추출 후 랜덤 선택
                frames = self.extract_frames_from_video(file_path)
                if len(frames) == 0:
                    image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                else:
                    # 랜덤 프레임 선택 (매 epoch마다 다른 프레임)
                    image = random.choice(frames)
                    image = self.process_image(image)
            else:
                # 이미지
                image = self.load_image(file_path)
                image = self.process_image(image)
            
            # Augmentation
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # 기본 변환
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                # Normalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = (image - mean) / std
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            image = torch.zeros(3, self.image_size, self.image_size)
        
        return image, torch.tensor(label, dtype=torch.float32)
