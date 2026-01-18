"""
최종 모델 학습 설정 - 모든 데이터 사용
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FinalTrainConfig:
    # Paths - 모든 데이터 사용 (Colab)
    image_real_dir: str = "/content/dataset/image/Train/Real"
    image_fake_dir: str = "/content/dataset/image/Train/Fake"
    video_real_dir: str = "/content/dataset/video/real"
    video_fake_dir: str = "/content/dataset/video/fake"
    
    output_dir: str = "./outputs_final"
    checkpoint_dir: str = "./checkpoints/model_4_final"
    
    # Model - 속도와 성능 균형 (T4 최적화)
    model_name: str = "convnext_small"  # base → small (50M 파라미터, 빠르면서 강력)
    pretrained: bool = True  # ImageNet 사전학습
    num_classes: int = 1
    
    # Training - 효율적 학습
    epochs: int = 15  # 작은 모델이라 더 많은 에폭 가능
    batch_size: int = 64  # 32 → 64 (GPU 활용도 ↑, 속도 ↑)
    num_workers: int = 4  # 2 → 4 (데이터 로딩 병렬화)
    
    # Optimizer - 빠른 수렴
    optimizer: str = "adamw"
    learning_rate: float = 5e-4  # 3e-4 → 5e-4 (빠른 수렴)
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 1  # 2 → 1 (warmup 줄임)
    
    # Loss
    loss_fn: str = "bce_with_logits"
    label_smoothing: float = 0.1  # 레이블 스무딩
    
    # Mixed Precision
    use_amp: bool = True
    gradient_clip: float = 1.0
    
    # Data - 효율적 샘플링
    image_size: int = 224
    face_detection: bool = True
    num_frames_per_video: int = 8  # 16 → 8 (속도 2배 ↑, 성능 유지)
    use_video: bool = True
    
    # Augmentation - 필수만 사용 (속도 최적화)
    random_resized_crop: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0  # 0.1 → 0.0 (제거)
    rotation_limit: int = 10  # 15 → 10
    jpeg_quality_range: tuple = (40, 100)  # (30, 100) → (40, 100)
    gaussian_blur_prob: float = 0.2  # 0.3 → 0.2
    gaussian_noise_prob: float = 0.2
    color_jitter: bool = True
    brightness_contrast_prob: float = 0.2  # 0.3 → 0.2
    
    # Validation
    val_split: float = 0.1  # 10% validation
    
    # Data Sampling - 핵심 최적화! (속도 40% ↑)
    max_samples_per_class: int = 30000  # 50000 → 30000
    
    # Seed
    seed: int = 42
    
    # Device
    device: str = "cuda"
    
    # Early Stopping
    patience: int = 5
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
