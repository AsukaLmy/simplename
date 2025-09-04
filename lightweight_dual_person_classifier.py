import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_person_classifier import DualPersonInteractionClassifier


class LightweightDualPersonClassifier(DualPersonInteractionClassifier):
    """
    Lightweight version of DualPersonInteractionClassifier optimized for small sample training
    
    Key optimizations:
    1. Simplified classifier architecture
    2. Reduced feature dimensions
    3. Built-in regularization
    4. Optimized for 1000-5000 samples per epoch
    """
    
    def __init__(self, backbone_name='mobilenet', pretrained=True, num_interaction_classes=5, 
                 fusion_method='concat', shared_backbone=True, 
                 freeze_backbone=True, lightweight_classifier=True):
        """
        Args:
            freeze_backbone: Whether to freeze backbone parameters
            lightweight_classifier: Use simplified classifier architecture
        """
        super().__init__(backbone_name, pretrained, num_interaction_classes, 
                        fusion_method, shared_backbone)
        
        self.freeze_backbone = freeze_backbone
        self.lightweight_classifier = lightweight_classifier
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone_A.parameters():
                param.requires_grad = False
            if not shared_backbone:
                for param in self.backbone_B.parameters():
                    param.requires_grad = False
            print(f"Backbone frozen: {freeze_backbone}")
        
        # Replace heavy classifiers with lightweight versions
        if lightweight_classifier:
            self._build_lightweight_classifiers()
            print("Using lightweight classifiers")
    
    def _build_lightweight_classifiers(self):
        """Build simplified classifiers with fewer parameters"""
        
        # Simplified Stage 1: Binary interaction detection
        self.stage1_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.fused_feature_dim, 256),  # Reduced from 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),  # Additional reduction
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification
        )
        
        # Simplified Stage 2: Interaction type classification
        self.stage2_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.fused_feature_dim, 256),  # Reduced from 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),  # Additional reduction
            nn.ReLU(),
            nn.Linear(64, self.num_interaction_classes)
        )
        
        # Calculate new parameter count
        stage1_params = sum(p.numel() for p in self.stage1_classifier.parameters())
        stage2_params = sum(p.numel() for p in self.stage2_classifier.parameters())
        print(f"Lightweight classifiers: Stage1={stage1_params:,}, Stage2={stage2_params:,} params")


class UltraLightweightDualPersonClassifier(nn.Module):
    """
    Ultra-lightweight classifier for very small datasets (< 2000 samples)
    Uses minimal parameters to prevent overfitting
    """
    
    def __init__(self, backbone_name='mobilenet', pretrained=True, 
                 fusion_method='concat', shared_backbone=True):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.shared_backbone = shared_backbone
        
        # Use pretrained backbone and freeze all layers
        if backbone_name == 'mobilenet':
            import torchvision.models as models
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            self.backbone_A = mobilenet.features
            if shared_backbone:
                self.backbone_B = self.backbone_A
            else:
                mobilenet_B = models.mobilenet_v2(pretrained=pretrained)
                self.backbone_B = mobilenet_B.features
            self.feature_dim = 1280
        
        # Freeze all backbone parameters
        for param in self.backbone_A.parameters():
            param.requires_grad = False
        if not shared_backbone:
            for param in self.backbone_B.parameters():
                param.requires_grad = False
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine fusion feature dimension
        if fusion_method == 'concat':
            self.fused_feature_dim = self.feature_dim * 2 if not shared_backbone else self.feature_dim * 2
        else:
            self.fused_feature_dim = self.feature_dim
            
        # Ultra-simple classifier - minimal parameters
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # High dropout for regularization
            nn.Linear(self.fused_feature_dim, 32),  # Very small hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)  # Binary output
        )
        
        print(f"Ultra-lightweight classifier initialized:")
        print(f"  Backbone: {backbone_name} (frozen)")
        print(f"  Shared backbone: {shared_backbone}")
        print(f"  Fusion method: {fusion_method}")
        print(f"  Feature dimensions: {self.feature_dim} -> {self.fused_feature_dim} -> 32 -> 2")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    def extract_person_features(self, person_images, backbone):
        """Extract features for a batch of person images"""
        features = backbone(person_images)
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        return pooled_features
    
    def fuse_features(self, features_A, features_B):
        """Fuse features from two persons"""
        if self.fusion_method == 'concat':
            fused_features = torch.cat([features_A, features_B], dim=1)
        elif self.fusion_method == 'add':
            fused_features = features_A + features_B
        elif self.fusion_method == 'subtract':
            fused_features = torch.abs(features_A - features_B)
        elif self.fusion_method == 'multiply':
            fused_features = features_A * features_B
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_features
    
    def forward(self, person_A_images, person_B_images):
        """Forward pass for binary classification only"""
        features_A = self.extract_person_features(person_A_images, self.backbone_A)
        features_B = self.extract_person_features(person_B_images, self.backbone_B)
        
        fused_features = self.fuse_features(features_A, features_B)
        output = self.classifier(fused_features)
        
        return {
            'stage1': output,
            'features_A': features_A,
            'features_B': features_B,
            'fused_features': fused_features
        }


def get_optimized_model_for_sample_size(sample_size, backbone_name='mobilenet'):
    """
    Get the optimal model configuration based on training sample size
    
    Args:
        sample_size: Number of training samples per epoch
        backbone_name: Backbone network to use
    
    Returns:
        Configured model and recommended training settings
    """
    
    if sample_size < 1000:
        # Ultra-lightweight for very small datasets
        model = UltraLightweightDualPersonClassifier(
            backbone_name=backbone_name,
            pretrained=True,
            fusion_method='concat',
            shared_backbone=True
        )
        training_config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-2,  # Strong regularization
            'dropout': 0.5,
            'batch_size': 16,
            'scheduler': 'cosine',
            'early_stopping_patience': 10
        }
        
    elif sample_size < 5000:
        # Lightweight for small datasets
        model = LightweightDualPersonClassifier(
            backbone_name=backbone_name,
            pretrained=True,
            fusion_method='concat',
            shared_backbone=True,
            freeze_backbone=True,
            lightweight_classifier=True
        )
        training_config = {
            'learning_rate': 5e-4,
            'weight_decay': 5e-3,
            'dropout': 0.3,
            'batch_size': 32,
            'scheduler': 'step',
            'early_stopping_patience': 15
        }
        
    else:
        # Standard model for larger datasets
        model = DualPersonInteractionClassifier(
            backbone_name=backbone_name,
            pretrained=True,
            fusion_method='concat',
            shared_backbone=False
        )
        training_config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'dropout': 0.2,
            'batch_size': 64,
            'scheduler': 'step',
            'early_stopping_patience': 20
        }
    
    return model, training_config


if __name__ == '__main__':
    # Test different model sizes
    sample_sizes = [500, 1000, 2000, 5000, 10000]
    
    for sample_size in sample_sizes:
        print(f"\n=== Sample size: {sample_size} ===")
        model, config = get_optimized_model_for_sample_size(sample_size)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Recommended config: LR={config['learning_rate']}, WD={config['weight_decay']}")
        print(f"Parameters per sample: {trainable_params/sample_size:.1f}")