"""
비디오 전용 추론 코드 - 더 많은 프레임 사용

MP4 파일만 처리하여 높은 정확도 확보
"""
import torch
import torch.nn as nn
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import DeepfakeDetector
from face_detector import FaceDetector


class VideoInference:
    def __init__(
        self,
        model_path: str,
        model_name: str = "convnext_small",
        device: str = "cpu",
        use_face_detection: bool = True,
        image_size: int = 224,
        num_frames: int = 16  # 기본 8 → 16으로 증가
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_face_detection = use_face_detection
        self.image_size = image_size
        self.num_frames = num_frames
        
        # 얼굴 검출기
        if use_face_detection:
            self.face_detector = FaceDetector()
        else:
            self.face_detector = None
        
        # 모델 로드
        print(f"Loading model from {model_path}...")
        self.model = DeepfakeDetector(
            model_name=model_name,
            pretrained=False,
            num_classes=1
        )
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully")
        
        # 변환
        self.transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def extract_frames(self, video_path: str):
        """비디오에서 프레임 추출 (더 많은 프레임)"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # 균등 샘플링
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 얼굴 검출
            if self.use_face_detection and self.face_detector:
                frame = self.face_detector.crop_face_with_fallback(
                    frame,
                    target_size=(self.image_size, self.image_size)
                )
            else:
                # 중앙 크롭
                h, w = frame.shape[:2]
                size = min(h, w)
                y1 = (h - size) // 2
                x1 = (w - size) // 2
                frame = frame[y1:y1+size, x1:x1+size]
                frame = cv2.resize(frame, (self.image_size, self.image_size))
            
            # 변환
            if self.transform:
                frame = self.transform(image=frame)['image']
            
            frames.append(frame)
        
        cap.release()
        return frames
    
    def predict_video(self, video_path: str):
        """비디오 예측 (평균값 사용)"""
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            return 0.5  # 기본값
        
        # 배치로 변환
        frames_tensor = torch.stack(frames).to(self.device)
        
        with torch.no_grad():
            logits = self.model(frames_tensor)
            probs = torch.sigmoid(logits)
        
        # 평균값 사용 (중앙값보다 안정적)
        avg_prob = probs.mean().item()
        
        return avg_prob


def inference_videos(
    model_path: str,
    test_dir: str,
    output_csv: str,
    model_name: str = "convnext_small",
    image_size: int = 224,
    use_face_detection: bool = True,
    num_frames: int = 16,  # ← 프레임 수 증가
    device: str = "cpu"
):
    """
    비디오 전용 추론
    
    Args:
        num_frames: 비디오당 추출할 프레임 수 (많을수록 정확하지만 느림)
    """
    
    # 비디오 추론기 생성
    inferencer = VideoInference(
        model_path=model_path,
        model_name=model_name,
        device=device,
        use_face_detection=use_face_detection,
        image_size=image_size,
        num_frames=num_frames
    )
    
    # 비디오 파일만 찾기
    test_path = Path(test_dir)
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_exts:
        video_files.extend(list(test_path.glob(f"*{ext}")))
        video_files.extend(list(test_path.glob(f"*{ext.upper()}")))
    
    video_files = sorted(video_files)
    
    print(f"\n비디오 파일: {len(video_files)}개")
    print(f"프레임 수: {num_frames}개/비디오")
    print(f"얼굴 검출: {use_face_detection}")
    print("-" * 60)
    
    # 추론
    results = []
    total = len(video_files)
    
    print(f"\n추론 시작: 총 {total}개 비디오")
    print("-" * 60)
    
    for idx, video_path in enumerate(video_files, 1):
        prob = inferencer.predict_video(str(video_path))
        
        results.append({
            'filename': video_path.name,
            'probability': prob
        })
        
        # 매 파일마다 진행 상황 출력
        progress = idx / total * 100
        print(f"[{idx}/{total}] ({progress:.1f}%) {video_path.name}: {prob:.4f}")
    
    print("-" * 60)
    
    # CSV 저장
    df = pd.DataFrame(results)
    df.columns = ['filename', 'prob']
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ 결과 저장: {output_csv}")
    print(f"총 예측: {len(df)}개")
    
    return df
