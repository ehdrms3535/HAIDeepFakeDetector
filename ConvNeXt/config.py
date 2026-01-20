"""
파인튜닝 설정

모델 경로 변경 시: pretrained_model 경로 수정
데이터 경로 변경 시: image_real_dir, image_fake_dir, video_real_dir, video_fake_dir 수정
출력 경로 변경 시: output_dir, checkpoint_dir 수정
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # 데이터 경로
    image_real_dir: str = "../dataset/image/Train/Real"
    image_fake_dir: str = "../dataset/image/Train/Fake"
    video_real_dir: str = "../dataset/video/real"
    video_fake_dir: str = "../dataset/video/fake"
    
    # ========================================
    # 모델 경로 설정 (여기서 모델 위치 변경)
    # ========================================
    # 사전학습 모델 경로 (변경 시 이 경로 수정)
    pretrained_model: str = "./checkpoints/model/best_model.pt"  # ← 불러올 모델 경로
    
    # 출력 경로
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints/model_1"  # ← 저장될 체크포인트 경로
    
    # 모델 설정
    model_name: str = "convnext_small"
    pretrained: bool = False  # 사전학습 모델 로드하므로 False
    num_classes: int = 1
    
    # ========================================
    # 학습 설정 (여기서 학습 조절)
    # ========================================
    epochs: int = 1  # ← 에포크 수: 1 epoch씩 점진적 학습
    batch_size: int = 4  # ← 배치 크기: 한번에 처리할 이미지 수
    num_workers: int = 0
    
    # 옵티마이저 설정
    optimizer: str = "adamw"
    learning_rate: float = 1e-5  # 파인튜닝용 낮은 learning rate
    weight_decay: float = 1e-4
    
    # 스케줄러 설정
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    
    # 손실 함수 설정
    loss_fn: str = "bce_with_logits"
    label_smoothing: float = 0.1
    
    # Mixed Precision 설정
    use_amp: bool = False
    gradient_clip: float = 1.0
    
    # 데이터 전처리 설정
    image_size: int = 224
    face_detection: bool = True  # 얼굴 검출 활성화
    num_frames_per_video: int = 8  # ← 비디오당 프레임 수: 비디오 1개에서 추출할 프레임 개수
    
    # 데이터 증강 설정
    random_resized_crop: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    rotation_limit: int = 10
    jpeg_quality_range: tuple = (70, 100)
    gaussian_blur_prob: float = 0.1
    gaussian_noise_prob: float = 0.1
    color_jitter: bool = True
    brightness_contrast_prob: float = 0.2
    
    # 검증 설정
    val_split: float = 0.1
    
    # ========================================
    # 데이터 샘플링 설정 (여기서 데이터 양 조절)
    # ========================================
    max_samples_per_class: int = 10  # ← 이미지 개수: 클래스당 10개 = Real 10개 + Fake 10개
    max_video_samples_per_class: int = 1  # ← 비디오 개수: 클래스당 1개 = Real 1개 + Fake 1개
    sample_offset: int = 0  # ← 시작 위치: 0부터 시작 (다음 학습시 10, 20, 30... 으로 변경)
    use_video: bool = True  # ← 비디오 사용 여부: True=사용, False=이미지만
    
    # 기타 설정
    seed: int = 42
    device: str = "cpu"
    patience: int = 3
    
    # ========================================
    # 추론 설정 (여기서 추론 경로 변경)
    # ========================================
    inference_model_path: str = "./checkpoints/step1/best_model.pt"  # ← 추론에 사용할 모델 경로
    test_data_dir: str = "./open/test_data"  # ← 테스트 데이터 경로
    submission_dir: str = "./submissions/step1"  # ← 제출 파일 저장 경로
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
