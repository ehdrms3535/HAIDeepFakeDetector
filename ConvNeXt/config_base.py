"""
대규모 학습 설정 - ConvNeXt Base 모델
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseConfig:
    # Paths
    image_real_dir: str = "../dataset/image/Train/Real"
    image_fake_dir: str = "../dataset/image/Train/Fake"
    video_real_dir: str = "../dataset/video/real"
    video_fake_dir: str = "../dataset/video/fake"
    
    # 사전학습 모델 사용
    pretrained_model: str = None  # 처음부터 학습
    
    output_dir: str = "./outputs_base"
    checkpoint_dir: str = "./checkpoints/model_base"
    
    # Model - ConvNeXt Base (더 강력)
    model_name: str = "convnext_base"  # Small(50M) -> Base(89M)
    pretrained: bool = True  # ImageNet 사전학습 가중치 사용
    num_classes: int = 1
    
    # Training - 순차적으로 1 epoch씩
    epochs: int = 1
    batch_size: int = 2  # Base 모델은 더 크므로 batch size 줄임
    num_workers: int = 0
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
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
    
    # Data - 대규모
    image_size: int = 224
    face_detection: bool = False
    num_frames_per_video: int = 8
    use_video: bool = True  # 비디오도 사용
    
    # Augmentation
    random_resized_crop: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    rotation_limit: int = 10
    jpeg_quality_range: tuple = (70, 100)
    gaussian_blur_prob: float = 0.1
    gaussian_noise_prob: float = 0.1
    color_jitter: bool = True
    brightness_contrast_prob: float = 0.2
    
    # Validation
    val_split: float = 0.1
    
    # Data Sampling - 대규모 (10000개)
    max_samples_per_class: int = 5000  # 클래스당 5000개 = 총 10000개
    sample_offset: int = 0  # 처음부터
    
    # Seed
    seed: int = 42
    
    # Device
    device: str = "cpu"
    
    # Early Stopping
    patience: int = 3
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
