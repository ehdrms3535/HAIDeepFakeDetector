"""
딥페이크 데이터셋

데이터 경로 변경 시: __init__의 image_real_dir, image_fake_dir, video_real_dir, video_fake_dir 수정
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


class DeepfakeDataset(Dataset):
    """딥페이크 탐지 데이터셋"""
    
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
        max_video_samples_per_class: int = None,  # 비디오 샘플 수 추가
        sample_offset: int = 0
    ):
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.num_frames_per_video = num_frames_per_video
        self.image_size = image_size
        self.sample_offset = sample_offset
        self.max_video_samples_per_class = max_video_samples_per_class  # 비디오 샘플 수 저장
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
        
        # 3. 비디오 - REAL (샘플링 적용)
        video_real_count = 0
        video_real_path = Path(video_real_dir)
        if video_real_path.exists():
            video_real_files = []
            for ext in video_exts:
                video_real_files.extend(list(video_real_path.glob(f"*{ext}")))
            
            # 비디오 샘플링 (config에서 설정한 개수만큼)
            if self.max_video_samples_per_class:
                start = sample_offset
                end = start + self.max_video_samples_per_class
                video_real_files = sorted(video_real_files)[start:end]

            for file_path in video_real_files:
                self.samples.append((str(file_path), 0, 'video'))
                video_real_count += 1
        
        print(f"  Video REAL: {video_real_count}")
        
        # 4. 비디오 - FAKE (샘플링 적용)
        video_fake_count = 0
        video_fake_path = Path(video_fake_dir)
        if video_fake_path.exists():
            video_fake_files = []
            for ext in video_exts:
                video_fake_files.extend(list(video_fake_path.glob(f"*{ext}")))
            
            # 비디오 샘플링 (config에서 설정한 개수만큼)
            if self.max_video_samples_per_class:
                video_fake_files = sorted(video_fake_files)
                video_fake_files = video_fake_files[:self.max_video_samples_per_class]  # ← config.py의 max_video_samples_per_class 값 사용
            
            for file_path in video_fake_files:
                self.samples.append((str(file_path), 1, 'video'))
                video_fake_count += 1
        
        print(f"  Video FAKE: {video_fake_count}")
        
        print(f"\nTotal samples: {len(self.samples)}")
        print(f"  REAL: {real_count + video_real_count}")
        print(f"  FAKE: {fake_count + video_fake_count}")
        
        # 셔플
        random.shuffle(self.samples)
    
    def _get_face_detector(self):
        # 워커 프로세스 안에서 최초 1회만 생성
        if self.face_detector is None and self.use_face_detection:
            self.face_detector = FaceDetector()
        return self.face_detector

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
    
    def extract_one_frame_from_video(self, path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return None
            idx = random.randint(0, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        if self.use_face_detection:
            fd = self._get_face_detector()
            if fd is not None:
                # 현재 image는 RGB임 -> FaceDetector는 BGR 기준이므로 변환
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                bgr = fd.crop_face_with_fallback(
                    bgr,
                    target_size=(self.image_size, self.image_size)
                )
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return image

        # fallback (기존 유지)
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
                frame = self.extract_one_frame_from_video(file_path)
                if frame is None:
                    image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                else:
                    image = self.process_image(frame)
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
