"""
딥페이크 탐지 모델
"""
import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    """ConvNeXt 기반 딥페이크 탐지 모델"""
    
    def __init__(self, model_name='convnext_small', pretrained=True, num_classes=1):
        super().__init__()
        
        # ConvNeXt 백본
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # feature extractor로 사용
        )
        
        # Feature dimension
        if 'tiny' in model_name:
            feature_dim = 768
        elif 'small' in model_name:
            feature_dim = 768
        elif 'base' in model_name:
            feature_dim = 1024
        elif 'large' in model_name:
            feature_dim = 1536
        else:
            feature_dim = 768
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
