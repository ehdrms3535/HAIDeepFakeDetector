"""
파인튜닝 학습 코드

모델 경로 변경 시: config.py의 pretrained_model 수정
체크포인트 저장 경로 변경 시: config.py의 checkpoint_dir 수정
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import warnings
import random
warnings.filterwarnings('ignore')

from model import DeepfakeDetector
from dataset import DeepfakeDataset
from config import Config
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_transforms(config: Config):
    """데이터 증강 변환 생성"""
    train_transform = A.Compose([
        A.HorizontalFlip(p=config.horizontal_flip_prob),
        A.VerticalFlip(p=config.vertical_flip_prob),
        A.Rotate(limit=config.rotation_limit, p=0.3),
        A.ImageCompression(
            quality_lower=config.jpeg_quality_range[0],
            quality_upper=config.jpeg_quality_range[1],
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=config.gaussian_blur_prob),
        A.GaussNoise(var_limit=(10.0, 50.0), p=config.gaussian_noise_prob),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.3 if config.color_jitter else 0
        ),
        A.RandomBrightnessContrast(p=config.brightness_contrast_prob),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def agg_logits_fn(logits_bn1, agg="mean", topk_ratio=0.25):
    """
    logits_bn1: [B, N, 1]
    return: [B, 1]
    """
    if agg == "mean":
        return logits_bn1.mean(dim=1)
    elif agg == "median":
        return logits_bn1.median(dim=1).values
    elif agg == "topkmean":
        B, N, _ = logits_bn1.shape
        k = max(1, int(N * topk_ratio))
        topk_vals, _ = torch.topk(logits_bn1, k=k, dim=1)  # [B, k, 1]
        return topk_vals.mean(dim=1)
    else:
        raise ValueError(f"Unknown agg: {agg}")


def train_epoch(model, loader, criterion, optimizer, scaler, device, config):
    """1 에포크 학습 수행 (비디오: 프레임별 추론 후 집계)"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        # images: [B, N, C, H, W]  (이미지는 N=1)
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            if images.dim() == 4:
                images = images.unsqueeze(1)   # [B,1,C,H,W]

            B, N, C, H, W = images.shape
            x = images.view(B * N, C, H, W)     # [B*N, C, H, W]

            logits = model(x)                   # [B*N, 1]
            logits = logits.view(B, N, 1)       # [B, N, 1]

            # 집계 (config에 추가해두면 편함)
            agg = getattr(config, "agg", "mean")
            topk_ratio = getattr(config, "topk_ratio", 0.25)

            agg_logits = agg_logits_fn(logits, agg=agg, topk_ratio=topk_ratio)  # [B,1]
            loss = criterion(agg_logits, labels)

        scaler.scale(loss).backward()

        if config.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)



def validate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)           # [B, N, C, H, W]
            labels = labels.to(device).unsqueeze(1)

            B, N, C, H, W = images.shape
            x = images.view(B*N, C, H, W)

            logits = model(x)                    # [B*N, 1]
            logits = logits.view(B, N, 1)        # [B, N, 1]

            agg = getattr(config, "agg", "mean")
            topk_ratio = getattr(config, "topk_ratio", 0.25)

            agg_logits = agg_logits_fn(
                logits, agg=agg, topk_ratio=topk_ratio
            )                                    # [B, 1]

            loss = criterion(agg_logits, labels)
            probs = torch.sigmoid(agg_logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            total_loss += loss.item()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    pred_labels = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, pred_labels, average='binary', zero_division=0
    )

    return {
        'loss': total_loss / len(loader),
        'auc': auc,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    config = Config()
    
    # 시드 설정
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # 디바이스 설정
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("=" * 60)
    
    # 데이터 변환 생성
    train_transform, val_transform = create_transforms(config)
    
    # 데이터셋 로드
    print("Loading dataset...")
    full_dataset = DeepfakeDataset(
        image_real_dir=config.image_real_dir,
        image_fake_dir=config.image_fake_dir,
        video_real_dir=config.video_real_dir,
        video_fake_dir=config.video_fake_dir,
        transform=None,
        use_face_detection=config.face_detection,
        num_frames_per_video=config.num_frames_per_video,
        image_size=config.image_size,
        max_samples_per_class=config.max_samples_per_class,
        max_video_samples_per_class=config.max_video_samples_per_class,  # 비디오 샘플 수 전달
        sample_offset=config.sample_offset
    )
    
    # Train/Val 분할
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # 변환 적용
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("=" * 60)
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )
    
    # 모델 생성
    print(f"\nCreating model: {config.model_name}")
    model = DeepfakeDetector(
        model_name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes
    )
    
    # 체크포인트 초기화
    start_epoch = 0
    best_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}
    
    # 체크포인트 로드 (이어서 학습)
    checkpoint_path = Path(config.checkpoint_dir) / "ckpt_last.pth"
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('best_auc', 0.0)
        history = checkpoint.get('history', history)
        print(f"✓ Resuming from epoch {start_epoch}, Best AUC: {best_auc:.4f}")
    elif Path(config.pretrained_model).exists():
        # 사전학습 모델 로드 (처음 시작)
        print(f"\nLoading pretrained model: {config.pretrained_model}")
        pretrained_state = torch.load(config.pretrained_model, map_location=device, weights_only=False)
        if 'model_state_dict' in pretrained_state:
            model.load_state_dict(pretrained_state['model_state_dict'])
        else:
            model.load_state_dict(pretrained_state)
        print("✓ Pretrained model loaded successfully")
    else:
        print(f"\nWarning: Pretrained model not found at {config.pretrained_model}")
        print("Starting from scratch with ImageNet weights")
    
    model = model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 손실 함수
    criterion = nn.BCEWithLogitsLoss()
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-7
    )
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    
    # 학습 루프
    patience_counter = 0
    
    for epoch in range(start_epoch, start_epoch + config.epochs):
        print(f"Epoch {epoch + 1}/{start_epoch + config.epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 학습
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )
        
        # 검증
        val_metrics = validate(model, val_loader, criterion, device,config)
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val Acc: {val_metrics['acc']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        
        # Best 모델 저장
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_path = Path(config.checkpoint_dir) / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"✓ Best model saved (AUC: {val_metrics['auc']:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 체크포인트 저장
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_auc': best_auc,
            'history': history
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # 스케줄러 스텝
        scheduler.step()
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping triggered (patience: {config.patience})")
            break
        
        print()
    
    # 최종 모델 저장
    final_path = Path(config.checkpoint_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    
    # 히스토리 저장
    history_path = Path(config.checkpoint_dir) / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print("Training completed!")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Models saved to: {config.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
