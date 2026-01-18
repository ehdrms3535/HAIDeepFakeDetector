"""
최종 모델 학습 - 모든 데이터 사용
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
warnings.filterwarnings('ignore')

from model import DeepfakeDetector
from dataset_final import FinalDeepfakeDataset
from config_final import FinalTrainConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_transforms(config: FinalTrainConfig):
    """강력한 데이터 증강"""
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


class LabelSmoothingBCELoss(nn.Module):
    """Label Smoothing을 적용한 BCE Loss"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # Label smoothing: 0 → smoothing, 1 → 1-smoothing
        target = target * (1 - self.smoothing) + self.smoothing / 2
        return self.bce(pred, target)


def train_epoch(model, loader, criterion, optimizer, scaler, device, config):
    """1 에포크 학습"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        
        if config.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            total_loss += loss.item()
    
    # Metrics
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
    config = FinalTrainConfig()
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 체크포인트 경로 확인
    resume_path = Path(config.checkpoint_dir) / 'ckpt_last.pth'
    start_epoch = 0
    best_auc = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # 기존 모델 로드 (pretrained_model이 있으면)
    load_pretrained = hasattr(config, 'pretrained_model') and Path(config.pretrained_model).exists()
    
    # Transforms
    train_transform, val_transform = create_transforms(config)
    
    # Dataset
    print("\n" + "="*60)
    full_dataset = FinalDeepfakeDataset(
        image_real_dir=config.image_real_dir,
        image_fake_dir=config.image_fake_dir,
        video_real_dir=config.video_real_dir,
        video_fake_dir=config.video_fake_dir,
        transform=None,
        use_face_detection=config.face_detection,
        num_frames_per_video=config.num_frames_per_video,
        image_size=config.image_size,
        max_samples_per_class=config.max_samples_per_class
    )
    
    # Train/Val split
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Transforms 적용
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("="*60)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model
    print(f"\nCreating model: {config.model_name}")
    model = DeepfakeDetector(
        model_name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes
    )
    
    # 기존 모델 로드 (fine-tuning)
    if load_pretrained:
        print(f"\nLoading pretrained model: {config.pretrained_model}")
        pretrained_checkpoint = torch.load(config.pretrained_model, map_location=device, weights_only=False)
        if 'model_state_dict' in pretrained_checkpoint:
            model.load_state_dict(pretrained_checkpoint['model_state_dict'])
        else:
            model.load_state_dict(pretrained_checkpoint)
        print("✓ Pretrained model loaded successfully")
    
    model = model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss & Optimizer
    if config.label_smoothing > 0:
        criterion = LabelSmoothingBCELoss(smoothing=config.label_smoothing)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs
    )
    
    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config.warmup_epochs
    )
    
    # Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    # 체크포인트에서 복구
    if resume_path.exists():
        print(f"\n{'='*60}")
        print(f"Found checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('best_auc', 0)
        history = checkpoint.get('history', history)
        
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best AUC so far: {best_auc:.4f}")
        print(f"{'='*60}\n")
    else:
        print(f"\nNo checkpoint found. Starting from scratch.\n")
    
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Scheduler
        if epoch < config.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # History
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
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['auc'],
                'val_acc': val_metrics['acc'],
            }, Path(config.checkpoint_dir) / 'best_model.pt')
            print(f"✓ Best model saved (AUC: {val_metrics['auc']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # 매 epoch마다 자동 저장 (Google Drive)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_auc': val_metrics['auc'],
            'val_acc': val_metrics['acc'],
            'val_loss': val_metrics['loss'],
            'train_loss': train_loss,
            'best_auc': best_auc,
            'history': history
        }
        
        # Last checkpoint 저장
        last_ckpt_path = Path(config.checkpoint_dir) / 'ckpt_last.pth'
        torch.save(checkpoint, last_ckpt_path)
        print(f"✓ Checkpoint saved: {last_ckpt_path}")
        
        # 매 N epoch마다 별도 저장 (복구용)
        if (epoch + 1) % 5 == 0:
            epoch_ckpt_path = Path(config.checkpoint_dir) / f'ckpt_epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_ckpt_path)
            print(f"✓ Epoch checkpoint saved: {epoch_ckpt_path}")
    
    # Save final model
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_metrics['auc'],
    }, Path(config.checkpoint_dir) / 'final_model.pt')
    
    # Save history
    with open(Path(config.checkpoint_dir) / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Models saved to: {config.checkpoint_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
