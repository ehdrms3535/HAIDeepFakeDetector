"""
적응형 프레임 추론 - 비디오 길이에 따라 프레임 수 자동 조정

규칙 준수:
- 비디오 길이 기반 프레임 수 결정 (신뢰도 기반 X)
- 단일 모델, 단일 추론 (TTA 없음)
- 프레임별 독립 추론 후 집계
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


def get_frame_count(video_path: str) -> int:
    """
    비디오 길이에 따라 프레임 수 결정 (신뢰도 기반 X)
    
    규칙 준수: 입력 특성(길이)에 따른 프레임 수 결정은 허용됨
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 긴 비디오 = 더 많은 프레임
    if total_frames > 300:
        return 32
    elif total_frames > 150:
        return 24
    else:
        return 16


class AdaptiveInference:
    """적응형 프레임 추론기"""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "convnext_base",
        device: str = "cpu",
        use_face_detection: bool = True,
        image_size: int = 224
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_face_detection = use_face_detection
        self.image_size = image_size
        
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
        print(f"✓ Model loaded on {self.device}")
        
        # 변환
        self.transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def extract_frames(self, video_path: str, num_frames: int):
        """비디오에서 프레임 추출 (적응형 프레임 수)"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # 균등 샘플링
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
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
    
    def process_image(self, image_path: str):
        """이미지 처리"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 얼굴 검출
        if self.use_face_detection and self.face_detector:
            img = self.face_detector.crop_face_with_fallback(
                img,
                target_size=(self.image_size, self.image_size)
            )
        else:
            # 중앙 크롭
            h, w = img.shape[:2]
            size = min(h, w)
            y1 = (h - size) // 2
            x1 = (w - size) // 2
            img = img[y1:y1+size, x1:x1+size]
            img = cv2.resize(img, (self.image_size, self.image_size))
        
        # 변환
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img
    
    def predict_video(self, video_path: str):
        """
        비디오 예측 (적응형 프레임 수)
        
        Returns:
            probability: fake일 확률 (0~1)
            num_frames: 사용된 프레임 수
        """
        # 비디오 길이에 따라 프레임 수 결정
        num_frames = get_frame_count(video_path)
        
        # 프레임 추출
        frames = self.extract_frames(video_path, num_frames)
        
        if len(frames) == 0:
            return 0.5, 0  # 기본값
        
        # 배치로 변환
        frames_tensor = torch.stack(frames).to(self.device)
        
        with torch.no_grad():
            logits = self.model(frames_tensor)
            probs = torch.sigmoid(logits)
        
        # 가중 집계: 평균 + 최대값 조합
        mean_prob = probs.mean().item()
        max_prob = probs.max().item()
        median_prob = probs.median().item()
        
        # 가중 평균 (안정적인 조합)
        final_prob = 0.5 * mean_prob + 0.3 * max_prob + 0.2 * median_prob
        
        return final_prob, len(frames)
    
    def predict_image(self, image_path: str):
        """이미지 예측"""
        img = self.process_image(image_path)
        
        if img is None:
            return 0.5
        
        img_tensor = img.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            prob = torch.sigmoid(logits).item()
        
        return prob


def run_adaptive_inference(
    model_path: str,
    test_dir: str,
    output_csv: str,
    model_name: str = "convnext_base",
    image_size: int = 224,
    use_face_detection: bool = True,
    device: str = "cpu"
):
    """
    적응형 프레임 추론 실행
    
    Args:
        model_path: 모델 가중치 경로
        test_dir: 테스트 데이터 디렉토리
        output_csv: 결과 CSV 파일 경로
        model_name: 모델 아키텍처
        image_size: 입력 이미지 크기
        use_face_detection: 얼굴 검출 사용 여부
        device: 디바이스 (cuda/cpu)
    """
    
    # 추론기 생성
    inferencer = AdaptiveInference(
        model_path=model_path,
        model_name=model_name,
        device=device,
        use_face_detection=use_face_detection,
        image_size=image_size
    )
    
    # 테스트 파일 찾기
    test_path = Path(test_dir)
    all_files = sorted(list(test_path.glob("*")))
    
    # 비디오/이미지 분류
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    image_exts = {'.jpg', '.jpeg', '.png', '.jfif', '.JPG', '.JPEG', '.PNG', '.JFIF'}
    
    video_files = [f for f in all_files if f.suffix in video_exts]
    image_files = [f for f in all_files if f.suffix in image_exts]
    
    print(f"\n{'='*60}")
    print(f"적응형 프레임 추론 시작")
    print(f"{'='*60}")
    print(f"총 파일: {len(all_files)}개")
    print(f"  - 비디오: {len(video_files)}개")
    print(f"  - 이미지: {len(image_files)}개")
    print(f"얼굴 검출: {use_face_detection}")
    print(f"디바이스: {device}")
    print(f"{'='*60}\n")
    
    # 추론 결과 저장
    results = []
    
    # 비디오 추론
    if len(video_files) > 0:
        print(f"[1/2] 비디오 추론 중...")
        print("-" * 60)
        
        for idx, video_path in enumerate(video_files, 1):
            prob, num_frames = inferencer.predict_video(str(video_path))
            
            results.append({
                'filename': video_path.name,
                'probability': prob
            })
            
            # 진행 상황 출력
            progress = idx / len(video_files) * 100
            print(f"  [{idx}/{len(video_files)}] ({progress:.1f}%) {video_path.name}: {prob:.4f} (frames: {num_frames})")
        
        print("-" * 60)
        print(f"✓ 비디오 추론 완료: {len(video_files)}개\n")
    
    # 이미지 추론
    if len(image_files) > 0:
        print(f"[2/2] 이미지 추론 중...")
        print("-" * 60)
        
        for idx, image_path in enumerate(image_files, 1):
            prob = inferencer.predict_image(str(image_path))
            
            results.append({
                'filename': image_path.name,
                'probability': prob
            })
            
            # 진행 상황 출력 (10개마다)
            if idx % 10 == 0 or idx == len(image_files):
                progress = idx / len(image_files) * 100
                print(f"  [{idx}/{len(image_files)}] ({progress:.1f}%) {image_path.name}: {prob:.4f}")
        
        print("-" * 60)
        print(f"✓ 이미지 추론 완료: {len(image_files)}개\n")
    
    # CSV 저장
    df = pd.DataFrame(results)
    df.columns = ['filename', 'prob']
    df = df.sort_values('filename').reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    
    print(f"{'='*60}")
    print(f"✓ 추론 완료!")
    print(f"{'='*60}")
    print(f"총 예측: {len(df)}개")
    print(f"결과 저장: {output_csv}")
    print(f"{'='*60}\n")
    
    # 통계 출력
    print("확률 분포:")
    print(f"  평균: {df['prob'].mean():.4f}")
    print(f"  중앙값: {df['prob'].median():.4f}")
    print(f"  최소: {df['prob'].min():.4f}")
    print(f"  최대: {df['prob'].max():.4f}")
    
    return df


if __name__ == "__main__":
    # 설정
    MODEL_PATH = "./checkpoints/step1/best_model.pt"
    TEST_DIR = "./open/test_data"
    OUTPUT_CSV = "./submissions/adaptive/submission.csv"
    MODEL_NAME = "convnext_small"  # step1 모델
    IMAGE_SIZE = 224
    USE_FACE_DETECTION = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 출력 디렉토리 생성
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    
    # 추론 실행
    df = run_adaptive_inference(
        model_path=MODEL_PATH,
        test_dir=TEST_DIR,
        output_csv=OUTPUT_CSV,
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        use_face_detection=USE_FACE_DETECTION,
        device=DEVICE
    )
    
    print("\n추론 완료!")
