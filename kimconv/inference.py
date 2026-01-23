"""
딥페이크 탐지 추론 코드

모델 경로 변경 시: inference() 함수의 model_path 파라미터 수정
테스트 데이터 경로 변경 시: test_dir 파라미터 수정
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
    """테스트 데이터셋 - 추론용"""
    def __init__(self, test_dir, transform=None, use_face_detection=True, num_frames=8, image_size=224):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.num_frames = num_frames
        self.image_size = image_size

        # ✅ 멀티프로세싱/피클링 이슈 피하려면 "워커에서 lazy init"이 가장 안전
        self.face_detector = None

        # ✅ 하위폴더까지 전부 파일 수집
        self.files = sorted([p for p in self.test_dir.rglob("*") if p.is_file()])

        # 비디오 확장자
        self.video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

    def _get_face_detector(self):
        if (self.face_detector is None) and self.use_face_detection:
            self.face_detector = FaceDetector()
        return self.face_detector

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_name = file_path.name

        if file_path.suffix.lower() in self.video_exts:
            frames = self._load_video(file_path)
        else:
            frames = self._load_image(file_path)

        return frames, file_name

    def _load_image(self, image_path: Path):
        """이미지 파일 로드 및 전처리"""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return torch.zeros(1, 3, self.image_size, self.image_size)

        # ✅ FaceDetector는 BGR 기준으로 동작
        if self.use_face_detection:
            fd = self._get_face_detector()
            if fd is not None:
                img_bgr = fd.crop_face_with_fallback(
                    img_bgr,
                    target_size=(self.image_size, self.image_size)
                )
            else:
                img_bgr = cv2.resize(img_bgr, (self.image_size, self.image_size))
        else:
            # 중앙 크롭 후 리사이즈 (BGR 그대로 처리)
            h, w = img_bgr.shape[:2]
            size = min(h, w)
            y1 = (h - size) // 2
            x1 = (w - size) // 2
            img_bgr = img_bgr[y1:y1+size, x1:x1+size]
            img_bgr = cv2.resize(img_bgr, (self.image_size, self.image_size))

        # RGB로 변환 (Albumentations는 보통 RGB 권장)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 유효성 확인
        if (not isinstance(img_rgb, np.ndarray)) or (len(img_rgb.shape) != 3) or (img_rgb.shape[0] == 0):
            return torch.zeros(1, 3, self.image_size, self.image_size)

        # Transform
        if self.transform:
            img_t = self.transform(image=img_rgb)['image']
        else:
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        return img_t.unsqueeze(0)  # [1, C, H, W]

    def _load_video(self, video_path: Path):
        """비디오에서 프레임 추출 및 전처리"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return torch.zeros(1, 3, self.image_size, self.image_size)

        # 균등 프레임 인덱스
        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

        frames = []
        fd = self._get_face_detector() if self.use_face_detection else None

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            # ✅ FaceDetector는 BGR 입력
            if fd is not None:
                face_bgr = fd.detect_face(frame_bgr)
                if face_bgr is not None:
                    frame_bgr = face_bgr

            # resize
            if isinstance(frame_bgr, np.ndarray) and frame_bgr.shape[0] > 0 and frame_bgr.shape[1] > 0:
                frame_bgr = cv2.resize(frame_bgr, (self.image_size, self.image_size))
            else:
                continue

            # RGB 변환 후 transform
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame_t = self.transform(image=frame_rgb)['image']
            else:
                frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

            frames.append(frame_t)

        cap.release()

        if len(frames) == 0:
            return torch.zeros(1, 3, self.image_size, self.image_size)

        return torch.stack(frames, dim=0)  # [N, C, H, W]


def inference(model_path, test_dir, output_csv, model_name="convnext_small", 
              image_size=224, batch_size=32, use_face_detection=True, 
              num_frames=8, device="cuda"):
    """
    추론 실행 함수
    
    Args:
        model_path: 모델 가중치 파일 경로 (.pt 파일)
        test_dir: 테스트 데이터 디렉토리 경로
        output_csv: 결과 저장 CSV 파일 경로
        model_name: 모델 아키텍처 이름
        image_size: 입력 이미지 크기
        batch_size: 배치 크기 (사용 안함, 호환성 유지용)
        use_face_detection: 얼굴 검출 사용 여부
        num_frames: 비디오당 추출할 프레임 수
        device: 디바이스 (cuda/cpu)
    """
    # 디바이스 설정
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 변환 정의
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # 데이터셋 생성
    dataset = TestDataset(
        test_dir=test_dir,
        transform=transform,
        use_face_detection=use_face_detection,
        num_frames=num_frames,
        image_size=image_size
    )
    
    print(f"\nTotal test files: {len(dataset)}")
    
    # 모델 생성 및 로드
    model = DeepfakeDetector(model_name=model_name, pretrained=False, num_classes=1)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model_path}")
    
    # 추론 수행
    results = []
    total_files = len(dataset)
    
    print(f"\n추론 시작: 총 {total_files}개 파일")
    print("-" * 60)
    
    with torch.no_grad():
        for idx, (frames, file_name) in enumerate(tqdm(dataset, desc="Inference"), 1):
            frames = frames.to(device)  # [N, C, H, W]
            
            # 모델 추론
            logits = model(frames)  # [N, 1]
            probs = torch.sigmoid(logits)  # [N, 1]
            
            # 비디오는 중앙값, 이미지는 평균값 사용
            if probs.shape[0] > 1:  # 비디오 (여러 프레임)
                avg_prob = probs.median().item()
            else:  # 이미지 (단일 프레임)
                avg_prob = probs.mean().item()
            
            results.append({
                'filename': file_name,
                'probability': avg_prob
            })
            
            # 진행 상황 출력 (5개마다)
            if idx % 5 == 0:
                print(f"  [{idx}/{total_files}] {file_name}: {avg_prob:.4f}")
    
    # CSV 저장
    df = pd.DataFrame(results)
    df.columns = ['filename', 'prob']  # 제출 양식에 맞게
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Results saved: {output_csv}")
    print(f"Total predictions: {len(df)}")
    
    return df
