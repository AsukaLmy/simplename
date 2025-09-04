import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_person_classifier import DualPersonInteractionClassifier


class EnhancedDualPersonClassifier(DualPersonInteractionClassifier):
    """
    Enhanced version of DualPersonInteractionClassifier optimized for large datasets (100K+ samples)
    
    Key enhancements:
    1. More sophisticated classifier architecture
    2. Better feature fusion strategies
    3. Optimized for full dataset training
    4. Added regularization techniques suitable for large data
    """
    
    def __init__(self, backbone_name='mobilenet', pretrained=True, num_interaction_classes=5, 
                 fusion_method='attention', shared_backbone=False, 
                 enhanced_classifier=True, dropout_rate=0.3):
        """
        Args:
            enhanced_classifier: Use more sophisticated classifier architecture
            dropout_rate: Dropout rate for regularization
        """
        # Initialize parent class
        super().__init__(backbone_name, pretrained, num_interaction_classes, 
                        fusion_method, shared_backbone)
        
        self.enhanced_classifier = enhanced_classifier
        self.dropout_rate = dropout_rate
        
        # Replace classifiers with enhanced versions for large data
        if enhanced_classifier:
            self._build_enhanced_classifiers()
            print("Using enhanced classifiers for large dataset")
        
        # Add feature normalization for stability
        self.feature_norm = nn.BatchNorm1d(self.fused_feature_dim)
        print(f"Added feature normalization layer")
    
    def _build_enhanced_classifiers(self):
        """Build enhanced classifiers with more capacity for large datasets"""
        
        # Enhanced Stage 1: Binary interaction detection with residual connections
        self.stage1_classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fused_feature_dim, 1024),  # Increased capacity
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        # Enhanced Stage 2: Interaction type classification
        self.stage2_classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fused_feature_dim, 1024),  # Increased capacity
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_interaction_classes)
        )
        
        # Calculate parameter count
        stage1_params = sum(p.numel() for p in self.stage1_classifier.parameters())
        stage2_params = sum(p.numel() for p in self.stage2_classifier.parameters())
        print(f"Enhanced classifiers: Stage1={stage1_params:,}, Stage2={stage2_params:,} params")
    
    def forward(self, person_A_images, person_B_images, stage='both'):
        """Enhanced forward pass with feature normalization"""
        # Extract and fuse features (same as parent)
        features_A = self.extract_person_features(person_A_images, self.backbone_A)
        features_B = self.extract_person_features(person_B_images, self.backbone_B)
        fused_features = self.fuse_features(features_A, features_B)
        
        # Apply feature normalization for training stability
        if self.training:
            fused_features = self.feature_norm(fused_features)
        
        results = {
            'features_A': features_A,
            'features_B': features_B,
            'fused_features': fused_features
        }
        
        if stage in ['stage1', 'both']:
            stage1_output = self.stage1_classifier(fused_features)
            results['stage1'] = stage1_output
        
        if stage in ['stage2', 'both']:
            stage2_output = self.stage2_classifier(fused_features)
            results['stage2'] = stage2_output
            
        return results


class OptimalDualPersonClassifier(nn.Module):
    """
    Optimal configuration specifically designed for 600K sample training
    Balances model complexity with data availability
    """
    
    def __init__(self, backbone_name='mobilenet', pretrained=True, 
                 fusion_method='attention', dropout_rate=0.2):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.backbone_name = backbone_name
        
        # Initialize backbone - use separate backbones for maximum capacity
        if backbone_name == 'mobilenet':
            import torchvision.models as models
            mobilenet_A = models.mobilenet_v2(pretrained=pretrained)
            mobilenet_B = models.mobilenet_v2(pretrained=pretrained)
            self.backbone_A = mobilenet_A.features
            self.backbone_B = mobilenet_B.features
            self.feature_dim = 1280
        elif backbone_name == 'resnet50':
            import torchvision.models as models
            resnet_A = models.resnet50(pretrained=pretrained)
            resnet_B = models.resnet50(pretrained=pretrained)
            # Remove final FC layer and avgpool
            self.backbone_A = nn.Sequential(*list(resnet_A.children())[:-2])
            self.backbone_B = nn.Sequential(*list(resnet_B.children())[:-2])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Advanced attention fusion for large datasets
        if fusion_method == 'attention':
            self.attention_net = nn.Sequential(
                nn.Linear(self.feature_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)
            )
            self.fused_feature_dim = self.feature_dim
        elif fusion_method == 'concat':
            self.fused_feature_dim = self.feature_dim * 2
        else:
            self.fused_feature_dim = self.feature_dim
        
        # Feature preprocessing
        self.feature_norm = nn.BatchNorm1d(self.fused_feature_dim)
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Optimal classifier for large dataset
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # Binary classification
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Optimal Dual-Person Classifier initialized:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Fusion method: {fusion_method}")
        print(f"  Feature dimensions: {self.feature_dim} * 2 -> {self.fused_feature_dim}")
        print(f"  Total params: {total_params:,}, Trainable: {trainable_params:,}")
        print(f"  Optimal for 400K+ training samples")
    
    def extract_person_features(self, person_images, backbone):
        """Extract features for a batch of person images"""
        features = backbone(person_images)
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        return pooled_features
    
    def fuse_features(self, features_A, features_B):
        """Advanced feature fusion optimized for large datasets"""
        if self.fusion_method == 'attention':
            concat_features = torch.cat([features_A, features_B], dim=1)
            attention_weights = self.attention_net(concat_features)
            
            weighted_A = features_A * attention_weights[:, 0:1]
            weighted_B = features_B * attention_weights[:, 1:2]
            fused_features = weighted_A + weighted_B
            
        elif self.fusion_method == 'concat':
            fused_features = torch.cat([features_A, features_B], dim=1)
            
        elif self.fusion_method == 'add':
            fused_features = features_A + features_B
            
        elif self.fusion_method == 'multiply':
            fused_features = features_A * features_B
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_features
    
    def forward(self, person_A_images, person_B_images):
        """Forward pass optimized for large dataset training"""
        # Extract individual features
        features_A = self.extract_person_features(person_A_images, self.backbone_A)
        features_B = self.extract_person_features(person_B_images, self.backbone_B)
        
        # Fuse features
        fused_features = self.fuse_features(features_A, features_B)
        
        # Apply normalization and regularization
        if self.training:
            fused_features = self.feature_norm(fused_features)
        fused_features = self.feature_dropout(fused_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return {
            'stage1': output,
            'features_A': features_A,
            'features_B': features_B,
            'fused_features': fused_features
        }


def get_optimal_model_for_large_dataset(sample_size=400000, backbone='mobilenet'):
    """
    Get optimal model configuration for large datasets
    """
    
    if sample_size >= 300000:
        # Large dataset: Use full capacity model
        model = OptimalDualPersonClassifier(
            backbone_name=backbone,
            pretrained=True,
            fusion_method='attention',
            dropout_rate=0.2
        )
        
        training_config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'scheduler': 'cosine',
            'early_stopping_patience': 10,
            'gradient_clipping': 1.0
        }
        
    elif sample_size >= 100000:
        # Medium-large dataset: Enhanced model
        model = EnhancedDualPersonClassifier(
            backbone_name=backbone,
            pretrained=True,
            fusion_method='attention',
            shared_backbone=False,
            enhanced_classifier=True,
            dropout_rate=0.3
        )
        
        training_config = {
            'learning_rate': 5e-4,
            'weight_decay': 5e-4,
            'batch_size': 32,
            'scheduler': 'step',
            'early_stopping_patience': 15,
            'gradient_clipping': 1.0
        }
        
    else:
        # Default to original model for smaller datasets
        from dual_person_classifier import DualPersonInteractionClassifier
        model = DualPersonInteractionClassifier(
            backbone_name=backbone,
            pretrained=True,
            fusion_method='attention',
            shared_backbone=True
        )
        
        training_config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-3,
            'batch_size': 16,
            'scheduler': 'plateau',
            'early_stopping_patience': 20,
            'gradient_clipping': 0.5
        }
    
    return model, training_config


if __name__ == '__main__':
    # Test optimal configurations for different dataset sizes
    dataset_sizes = [50000, 150000, 400000, 600000]
    
    for size in dataset_sizes:
        print(f"\n=== Dataset size: {size:,} ===")
        model, config = get_optimal_model_for_large_dataset(size)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_per_sample = trainable_params / size
        
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Parameters per sample: {param_per_sample:.1f}")
        print(f"Recommended config: {config}")
        
        if param_per_sample < 0.1:
            print("✅ Excellent parameter/sample ratio")
        elif param_per_sample < 0.5:
            print("✅ Good parameter/sample ratio")
        else:
            print("⚠️ Consider reducing model complexity")