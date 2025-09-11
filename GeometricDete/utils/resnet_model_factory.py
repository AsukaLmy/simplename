#!/usr/bin/env python3
"""
ResNet Model Factory for Stage2 Behavior Classification
Creates appropriate ResNet-based models and components based on configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import os

# 导入ResNet相关组件
from configs.resnet_stage2_config import ResNetStage2Config
from models.resnet_stage2_classifier import ResNetRelationStage2Classifier, ResNetStage2Loss
from models.resnet_feature_extractors import ResNetRelationFeatureFusion


def create_resnet_stage2_model(config: ResNetStage2Config) -> nn.Module:
    """
    根据配置创建ResNet Stage2模型
    
    Args:
        config: ResNet Stage2配置对象
        
    Returns:
        nn.Module: 创建的ResNet Relation Network模型
    """
    # 获取特征维度
    person_feature_dim = config.get_person_feature_dim()
    spatial_feature_dim = config.get_spatial_feature_dim()
    
    # 创建ResNet Relation Network模型
    model = ResNetRelationStage2Classifier(
        person_feature_dim=person_feature_dim,
        spatial_feature_dim=spatial_feature_dim,
        hidden_dims=config.relation_hidden_dims,
        dropout=config.dropout,
        fusion_strategy=config.fusion_strategy,
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        crop_size=config.crop_size
    )
    
    print(f"✅ Created ResNet Relation Network:")
    print(f"   Backbone: {config.backbone_name}")
    print(f"   Person features: {person_feature_dim}D")
    print(f"   Spatial features: {spatial_feature_dim}D")
    print(f"   Fusion: {config.fusion_strategy}")
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"   Parameters: {model_info['trainable_params']:,}")
    
    return model


def create_resnet_stage2_loss(config: ResNetStage2Config) -> ResNetStage2Loss:
    """
    创建ResNet Stage2损失函数
    
    Args:
        config: ResNet Stage2配置对象
        
    Returns:
        ResNetStage2Loss: 损失函数
    """
    # 如果config中没有class_weights，提供均匀权重作为兜底
    if not hasattr(config, 'class_weights') or config.class_weights is None:
        config.class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
        print("⚠️ config.class_weights not found — using uniform fallback weights")
    
    criterion = ResNetStage2Loss(
        class_weights=config.class_weights
    )
    
    print(f"✅ Created ResNet Stage2 Loss: weights={config.class_weights}")
    return criterion


def create_resnet_optimizer(model: nn.Module, config: ResNetStage2Config) -> optim.Optimizer:
    """
    创建ResNet模型的优化器，支持不同学习率策略
    
    Args:
        model: ResNet模型
        config: 配置对象
        
    Returns:
        torch.optim.Optimizer: 优化器
    """
    # 分离ResNet backbone参数和其他参数，使用不同学习率
    backbone_params = []
    other_params = []

    # prefer to detect explicit backbone attribute on model
    model_for_iter = model.module if hasattr(model, 'module') else model
    if hasattr(model_for_iter, 'backbone'):
        for name, param in model_for_iter.backbone.named_parameters():
            if param is not None:
                backbone_params.append(param)
        # other params are model parameters minus backbone
        backbone_param_ids = {id(p) for p in backbone_params}
        for name, param in model_for_iter.named_parameters():
            if id(param) in backbone_param_ids:
                continue
            other_params.append(param)
    else:
        # fallback to name-based heuristic
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
    
    # 设置参数组：backbone使用较低学习率
    param_groups = []
    
    if backbone_params:
        backbone_lr = config.learning_rate * 0.1  # Backbone用1/10的学习率
        if getattr(config, 'freeze_backbone', False):
            # Freeze backbone parameters and remove from optimizer groups
            for p in backbone_params:
                p.requires_grad = False
            backbone_params = []
            print("   Backbone frozen via config.freeze_backbone; not included in optimizer param groups")
        else:
            param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })
        print(f"   Backbone params: {sum(p.numel() for p in backbone_params):,}, lr={backbone_lr}")
    
    if other_params:
        param_groups.append({
            'params': other_params, 
            'lr': config.learning_rate,
            'name': 'classifier'
        })
        print(f"   Classifier params: {sum(p.numel() for p in other_params):,}, lr={config.learning_rate}")
    
    # 创建优化器
    if config.optimizer == 'adam':
        optimizer = optim.Adam(
            param_groups if param_groups else model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(
            param_groups if param_groups else model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            param_groups if param_groups else model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    print(f"✅ Created {config.optimizer} optimizer with differential learning rates")
    return optimizer


def create_resnet_scheduler(optimizer: optim.Optimizer, config: ResNetStage2Config) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    创建ResNet模型的学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置对象
        
    Returns:
        Optional[_LRScheduler]: 学习率调度器
    """
    if config.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=0.1
        )
    elif config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
    elif config.scheduler == 'warmup_cosine':
        # 实现warmup + cosine scheduler
        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                return float(epoch) / float(max(1, config.warmup_epochs))
            else:
                progress = float(epoch - config.warmup_epochs) / float(max(1, config.epochs - config.warmup_epochs))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    if scheduler:
        print(f"✅ Created {config.scheduler} scheduler")
    else:
        print("✅ No scheduler used")
        
    return scheduler


def create_resnet_training_setup(config: ResNetStage2Config, device: torch.device) -> Tuple:
    """
    创建完整的ResNet训练设置
    
    Args:
        config: 配置对象
        device: 设备
        
    Returns:
        Tuple: (model, criterion, optimizer, scheduler)
    """
    print(f"\n🏗️ Creating ResNet training setup for {config.backbone_name}...")
    
    # 创建模型并移动到设备
    model = create_resnet_stage2_model(config).to(device)
    
    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)
        print(f"   Using DataParallel on {torch.cuda.device_count()} GPUs")
    
    # 创建损失函数
    criterion = create_resnet_stage2_loss(config).to(device)
    
    # 创建优化器（支持差分学习率）
    optimizer = create_resnet_optimizer(model, config)
    
    # 创建调度器
    scheduler = create_resnet_scheduler(optimizer, config)
    
    print(f"✅ ResNet training setup completed on {device}")
    
    return model, criterion, optimizer, scheduler


class ResNetModelCheckpointManager:
    """ResNet模型检查点管理器"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler],
                       epoch: int, metrics: dict, filename: str, config: ResNetStage2Config):
        """
        保存ResNet模型检查点
        """
        # 处理DataParallel模型
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': {
                'backbone_name': config.backbone_name,
                'visual_feature_dim': config.visual_feature_dim,
                'relation_hidden_dims': config.relation_hidden_dims,
                'fusion_strategy': config.fusion_strategy,
                'use_geometric': config.use_geometric,
                'use_scene_context': config.use_scene_context,
            },
            'model_info': model.module.get_model_info() if hasattr(model, 'module') else model.get_model_info()
        }
        
        filepath = os.path.join(self.save_dir, f"{filename}.pth")
        torch.save(checkpoint, filepath)
        print(f"💾 ResNet checkpoint saved: {filename}.pth")
    
    def load_checkpoint(self, filepath: str, model: nn.Module,
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> dict:
        """
        加载ResNet模型检查点
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 处理DataParallel模型
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📂 ResNet checkpoint loaded: {filepath}")
        print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   Config: {checkpoint.get('config', {})}")
        print(f"   Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint


if __name__ == '__main__':
    # 测试ResNet模型工厂
    print("Testing ResNet Model Factory...")
    
    from configs.resnet_stage2_config import get_resnet18_config
    
    # 创建测试配置
    config = get_resnet18_config(
        backbone_name='resnet18',
        visual_feature_dim=256,
        relation_hidden_dims=[512, 256, 128],
        dropout=0.3,
        batch_size=8
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试完整训练设置
    model, criterion, optimizer, scheduler = create_resnet_training_setup(config, device)
    
    print(f"\nTesting forward pass...")
    batch_size = 4
    
    # 创建测试数据
    person_A_features = torch.randn(batch_size, config.visual_feature_dim).to(device)
    person_B_features = torch.randn(batch_size, config.visual_feature_dim).to(device)
    spatial_features = torch.randn(batch_size, config.get_spatial_feature_dim()).to(device)
    targets = torch.randint(0, 3, (batch_size,)).to(device)
    
    # 前向传播
    with torch.no_grad():
        logits = model(person_A_features, person_B_features, spatial_features)
        loss, loss_dict = criterion(logits, targets)
    
    print(f"Input shapes:")
    print(f"  Person A: {person_A_features.shape}")
    print(f"  Person B: {person_B_features.shape}")  
    print(f"  Spatial: {spatial_features.shape}")
    print(f"Output:")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss details: {loss_dict}")
    
    # 测试检查点管理器
    print(f"\nTesting checkpoint manager...")
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = ResNetModelCheckpointManager(temp_dir)
        
        # 保存检查点
        test_metrics = {'val_accuracy': 0.85, 'val_mpca': 0.82}
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler,
            epoch=10, metrics=test_metrics,
            filename='test_resnet_checkpoint', config=config
        )
        
        # 加载检查点
        checkpoint_path = os.path.join(temp_dir, 'test_resnet_checkpoint.pth')
        loaded_checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
    
    print("\n✅ ResNet model factory test completed!")