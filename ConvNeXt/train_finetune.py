"""
CPU Fine-tuning - model_4 기반 추가 학습
train_final.py를 복사해서 config만 변경
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
from config_finetune import FineTuneConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2

# train_final.py의 함수들 import
from train_final import create_transforms, LabelSmoothingBCELoss, train_epoch, validate

def main():
    config = FineTuneConfig()  # FineTuneConfig 사용
    
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
        max_samples_per_class=config.max_samples_per_class,
        sample_offset=getattr(config, 'sample_offset', 0)  # offset 전달
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
    if config.warmup_epochs > 0:
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
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
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
        if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
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
        
        # 매 epoch마다 자동 저장
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
