#!/usr/bin/env python3
"""
ResNet-based Stage2 Configuration
Configuration for ResNet backbone with Relation Network
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ResNetStage2Config:
    """ResNet-based Stage2 behavior classification configuration"""
    
    # === Model Architecture ===
    model_type: str = "resnet_relation"  # Model type identifier
    backbone_name: str = "resnet18"  # ResNet architecture: resnet18, resnet34, resnet50
    visual_feature_dim: int = 128    # Visual feature dimension from ResNet
    
    # === Feature Configuration ===
    use_geometric: bool = True       # Use geometric spatial features  
    use_scene_context: bool = True   # Use scene context features
    
    # === ResNet Backbone Settings ===
    pretrained: bool = True          # Use ImageNet pretrained weights
    freeze_backbone: bool = False    # Whether to freeze backbone parameters
    freeze_blocks: int = 0           # Number of early residual blocks to freeze (0-4)
    crop_size: int = 112            # Person crop size for ResNet input
    padding_ratio: float = 0.2      # Padding ratio for person cropping
    
    # === Relation Network Settings ===
    fusion_strategy: str = "concat"  # Feature fusion: concat, bilinear, add
    relation_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    
    # === Training Configuration ===
    epochs: int = 50
    batch_size: int = 16            # Smaller batch size for ResNet memory usage
    learning_rate: float = 1e-3     # Lower LR for pretrained features
    weight_decay: float = 1e-5      # L2 regularization
    
    # === Optimization ===
    optimizer: str = "adam"         # Optimizer type
    scheduler: str = "cosine"       # LR scheduler: step, cosine, plateau, none
    step_size: int = 15            # For StepLR
    warmup_epochs: int = 3         # Warmup epochs for pretrained backbone
    
    # === Loss Configuration ===
    class_weights: Optional[Dict] = None  # Will be set by data loader
    mpca_weight: float = 0.1       # MPCA loss weight  
    acc_weight: float = 0.05       # Accuracy regularization weight
    
    # === Data Configuration ===
    data_path: str = "../dataset"
    frame_interval: int = 1         # Frame sampling interval
    num_workers: int = 4           # DataLoader workers
    use_oversampling: bool = True  # Use weighted sampling for class balance
    
    # === Early Stopping & Checkpointing ===
    early_stopping_patience: int = 15
    early_stopping_metric: str = "mpca"  # Metric for early stopping: mpca, accuracy
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints/resnet_stage2"
    
    # === Logging ===
    log_interval: int = 10         # Print frequency
    eval_interval: int = 1         # Validation frequency (epochs)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self.validate()
        self._setup_derived_configs()
    
    def validate(self):
        """Validate configuration parameters"""
        # Backbone validation
        supported_backbones = ['resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet']
        if self.backbone_name not in supported_backbones:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. Supported: {supported_backbones}"
            )
        
        # Fusion strategy validation
        supported_fusion = ['concat', 'bilinear', 'add']
        if self.fusion_strategy not in supported_fusion:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}. "
                           f"Supported: {supported_fusion}")
        
        # Feature configuration validation
        if not (self.use_geometric or self.use_scene_context):
            print("Warning: No spatial features enabled. Only using visual features.")
        
        # Training parameters validation
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not (0 <= self.dropout <= 1):
            raise ValueError("dropout must be in [0, 1]")
        
        # Dimension validation
        if self.visual_feature_dim < 1:
            raise ValueError("visual_feature_dim must be >= 1")
        if len(self.relation_hidden_dims) == 0:
            raise ValueError("relation_hidden_dims cannot be empty")

        # freeze_blocks validation
        if not isinstance(self.freeze_blocks, int) or not (0 <= self.freeze_blocks <= 4):
            raise ValueError("freeze_blocks must be an int in [0,4]")

        print("[SUCCESS] ResNet Stage2 config validation passed")
    
    def _setup_derived_configs(self):
        """Setup derived configuration values"""
        import os
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set default class weights if not provided
        if self.class_weights is None:
            self.class_weights = {0: 1.0, 1: 1.0, 2: 1.0}  # Will be updated by data loader
        
        print(f"[INFO] Checkpoint directory: {self.checkpoint_dir}")
    
    def get_person_feature_dim(self) -> int:
        """Get dimension of individual person features"""
        return self.visual_feature_dim

    def get_spatial_feature_dim(self) -> int:
        """Get dimension of spatial relation features"""
        dim = 0
        if self.use_geometric:
            dim += 7  # Geometric features
        if self.use_scene_context:
            dim += 1  # Scene context
        return dim
    
    def get_model_info(self) -> Dict:
        """Get model architecture information"""
        return {
            'model_type': self.model_type,
            'backbone': self.backbone_name,
            'visual_feature_dim': self.visual_feature_dim,
            'person_feature_dim': self.get_person_feature_dim(),
            'spatial_feature_dim': self.get_spatial_feature_dim(),
            'fusion_strategy': self.fusion_strategy,
            'hidden_dims': self.relation_hidden_dims,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'freeze_blocks': self.freeze_blocks,
        }
    
    def print_config(self):
        """Print configuration summary"""
        print("=" * 60)
        print("RESNET STAGE2 CONFIGURATION")
        print("=" * 60)
        
        print(f"Model Architecture:")
        print(f"  Type: {self.model_type}")
        print(f"  Backbone: {self.backbone_name}")
        print(f"  Visual features: {self.visual_feature_dim}D")
        print(f"  Spatial features: {self.get_spatial_feature_dim()}D")
        print(f"  Fusion strategy: {self.fusion_strategy}")
        
        print(f"\nBackbone Settings:")
        print(f"  Pretrained: {self.pretrained}")
        print(f"  Freeze backbone: {self.freeze_backbone}")
        print(f"  Crop size: {self.crop_size}x{self.crop_size}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Scheduler: {self.scheduler}")
        
        print(f"\nFeatures:")
        print(f"  Geometric: {self.use_geometric}")
        print(f"  Scene context: {self.use_scene_context}")
        
        print(f"\nData:")
        print(f"  Data path: {self.data_path}")
        print(f"  Frame interval: {self.frame_interval}")
        print(f"  Use oversampling: {self.use_oversampling}")
        
        print("=" * 60)


# 预定义配置
def get_resnet18_config(**kwargs) -> ResNetStage2Config:
    """Get ResNet18 configuration"""
    config = ResNetStage2Config(
        backbone_name="resnet18",
        visual_feature_dim=128,
        relation_hidden_dims=[128, 64, 32],
        batch_size=16,
        learning_rate=1e-4
    )
    
    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")
    
    return config


def get_resnet50_config(**kwargs) -> ResNetStage2Config:
    """Get ResNet50 configuration (more powerful but slower)"""
    config = ResNetStage2Config(
        backbone_name="resnet50",
        visual_feature_dim=512,
        relation_hidden_dims=[1024, 512, 256],
        batch_size=8,  # Smaller batch size for larger model
        learning_rate=5e-5  # Lower LR for larger model
    )

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_vgg16_config(**kwargs) -> ResNetStage2Config:
    """Get VGG16 configuration"""
    config = ResNetStage2Config(
        backbone_name="vgg16",
        visual_feature_dim=512,  # VGG16输出512维特征
        relation_hidden_dims=[512, 256, 128],
        batch_size=8,   # VGG需要更多显存
        learning_rate=5e-5  # 较低学习率
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_vgg19_config(**kwargs) -> ResNetStage2Config:
    """Get VGG19 configuration"""
    config = ResNetStage2Config(
        backbone_name="vgg19",
        visual_feature_dim=512,
        relation_hidden_dims=[512, 256, 128],
        batch_size=6,   # 更小批次
        learning_rate=5e-5
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def get_alexnet_config(**kwargs) -> ResNetStage2Config:
    """Get AlexNet configuration"""
    config = ResNetStage2Config(
        backbone_name="alexnet",
        visual_feature_dim=256,  # AlexNet输出256维特征
        relation_hidden_dims=[256, 128, 64],
        batch_size=32,  # AlexNet较轻量，可以用大批次
        learning_rate=1e-4
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")

    return config


def create_backbone_config(backbone_name: str, **kwargs) -> ResNetStage2Config:
    """根据backbone名称创建对应配置"""
    config_functions = {
        'resnet18': get_resnet18_config,
        'resnet34': lambda **kw: get_resnet18_config(backbone_name='resnet34', **kw),
        'resnet50': get_resnet50_config,
        'vgg11': lambda **kw: get_alexnet_config(backbone_name='vgg11', **kw),
        'vgg13': lambda **kw: get_alexnet_config(backbone_name='vgg13', **kw),
        'vgg16': get_vgg16_config,
        'vgg19': get_vgg19_config,
        'alexnet': get_alexnet_config
    }

    if backbone_name in config_functions:
        return config_functions[backbone_name](**kwargs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


def get_backbone_config(backbone_name: str, **kwargs) -> ResNetStage2Config:
    """获取backbone配置的便捷函数"""
    return create_backbone_config(backbone_name, **kwargs)


if __name__ == '__main__':
    # Test configurations
    print("Testing ResNet Stage2 Configurations...")
    
    print("\n1. Testing ResNet18 config:")
    config18 = get_resnet18_config()
    config18.print_config()
    
    print(f"\nModel info: {config18.get_model_info()}")
    
    print("\n2. Testing ResNet50 config:")
    config50 = get_resnet50_config(epochs=30)
    print(f"ResNet50 model info: {config50.get_model_info()}")
    
    print("\n3. Testing custom config:")
    custom_config = ResNetStage2Config(
        backbone_name="resnet34",
        visual_feature_dim=512,
        fusion_strategy="bilinear",
        freeze_backbone=True
    )
    print(f"Custom model info: {custom_config.get_model_info()}")
    
    print("\n✅ ResNet Stage2 configurations test completed!")