"""
Fine-tuning 설정 - Model 7 (Model 6 기반, 같은 데이터 범위)
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FineTuneConfig:
    # Paths - 로컬 데이터셋
    image_real_dir: str = "../dataset/image/Train/Real"
    image_fake_dir: str = "../dataset/image/Train/Fake"
    video_real_dir: str = "../dataset/video/real"
    video_fake_dir: str = "../dataset/video/fake"
    
    # 기존 모델 로드 - Model 6 사용!
    pretrained_model: str = "./checkpoints/model_6_finetune2/best_model.pt"
    
    output_dir: str = "./outputs_finetune7"
    checkpoint_dir: str = "./checkpoints/model_7_finetune"
    
    # Model
    model_name: str = "convnext_small"
    pretrained: bool = False
    num_classes: int = 1
    
    # Training - 1 epoch만
    epochs: int = 1
    batch_size: int = 4
    num_workers: int = 0
    
    # Optimizer - Fine-tuning용
    optimizer: str = "adamw"
    learning_rate: float = 5e-6  # 더 낮은 learning rate
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    
    # Loss
    loss_fn: str = "bce_with_logits"
    label_smoothing: float = 0.1
    
    # Mixed Precision
    use_amp: bool = False
    gradient_clip: float = 1.0
    
    # Data - Model 6과 같은 설정
    image_size: int = 224
    face_detection: bool = False  # Model 6은 얼굴 크롭 안함
    num_frames_per_video: int = 4
    use_video: bool = False
    
    # Augmentation - 최소화
    random_resized_crop: bool = False
    horizontal_flip_prob: float = 0.3
    vertical_flip_prob: float = 0.0
    rotation_limit: int = 0
    jpeg_quality_range: tuple = (90, 100)
    gaussian_blur_prob: float = 0.0
    gaussian_noise_prob: float = 0.0
    color_jitter: bool = False
    brightness_contrast_prob: float = 0.0
    
    # Validation
    val_split: float = 0.1
    
    # Data Sampling - Model 6과 같은 범위 (0~2000)
    max_samples_per_class: int = 1000  # 클래스당 1000개 = 총 2000개
    sample_offset: int = 0  # 0~1000번째 샘플
    
    # Seed
    seed: int = 42
    
    # Device
    device: str = "cpu"
    
    # Early Stopping
    patience: int = 2
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
