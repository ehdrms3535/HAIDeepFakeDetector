"""
ConvNeXt Base 모델 학습 - 대규모 데이터
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import random

from config_base import BaseConfig
from dataset_final import FinalDeepfakeDataset
from model import DeepfakeDetector
from train_final import get_transforms, train_epoch, validate


def main():
    # Config
    config = BaseConfig()
    
    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    print("=" * 60)
    
    # Transforms
    train_transform, val_transform = get_transforms(config)
    
    # Dataset
    full_dataset = FinalDeepfakeDataset(
        image_real_dir=config.image_real_dir,
        image_fake_dir=config.image_fake_dir,
        video_real_dir=config.video_real_dir,
        video_fake_dir=config.video_fake_dir,
        transform=train_transform,
        use_face_detection=config.face_detection,
        num_frames_per_video=config.num_frames_per_video,
        image_size=config.image_size,
        max_samples_per_class=config.max_samples_per_class,
        sample_offset=config.sample_offset
    )
    
    # Train/Val split
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Val dataset에 val transform 적용
    val_dataset.dataset.transform = val_transform
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("=" * 60)
    
    # DataLoader
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
    
    # Model
    print(f"\nCreating model: {config.model_name}")
    model = DeepfakeDetector(
        model_name=config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes
    )
    
    # 기존 모델 로드 (있으면)
    start_epoch = 0
    best_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': [], 'val_f1': []}
    
    checkpoint_path = Path(config.checkpoint_dir) / "ckpt_last.pth"
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('best_auc', 0.0)
        history = checkpoint.get('history', history)
        print(f"✓ Resuming from epoch {start_epoch}, Best AUC: {best_auc:.4f}")
    elif config.pretrained_model and Path(config.pretrained_model).exists():
        print(f"\nLoading pretrained model: {config.pretrained_model}")
        checkpoint = torch.load(config.pretrained_model, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Pretrained model loaded successfully")
    else:
        print("\nNo checkpoint found. Starting from scratch.")
    
    model = model.to(device)
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss
    if config.loss_fn == "bce_with_logits":
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0]).to(device)
        )
    else:
        criterion = nn.BCELoss()
    
    # Optimizer
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    # Scheduler
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    
    # Training loop
    patience_counter = 0
    
    for epoch in range(start_epoch, start_epoch + config.epochs):
        print(f"Epoch {epoch + 1}/{start_epoch + config.epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, config.gradient_clip, config.use_amp
        )
        
        # Validate
        val_loss, val_auc, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {val_auc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_path = Path(config.checkpoint_dir) / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"✓ Best model saved (AUC: {val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': best_auc,
            'history': history
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Scheduler step
        if scheduler and epoch >= config.warmup_epochs:
            scheduler.step()
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping triggered (patience: {config.patience})")
            break
        
        print()
    
    # Save history
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
