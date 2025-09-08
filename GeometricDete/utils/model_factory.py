#!/usr/bin/env python3
"""
Model Factory for Stage2 Behavior Classification
Creates appropriate models based on configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional

# 导入配置和模型类
from configs.stage2_config import Stage2Config
from models.stage2_classifier import BasicStage2Classifier, LSTMStage2Classifier, RelationStage2Classifier, Stage2Loss


def create_stage2_model(config: Stage2Config) -> nn.Module:
    """
    根据配置创建Stage2模型
    
    Args:
        config: Stage2配置对象
        
    Returns:
        nn.Module: 创建的模型
    """
    input_dim = config.get_input_dim()
    
    if config.temporal_mode == 'none':
        # Basic模式
        model = BasicStage2Classifier(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            use_attention=config.use_attention
        )
        print(f"✅ Created BasicStage2Classifier: {input_dim}D → 3 classes")
        
    elif config.temporal_mode == 'lstm':
        # LSTM模式
        model = LSTMStage2Classifier(
            feature_dim=input_dim,
            sequence_length=config.sequence_length,
            lstm_hidden_dim=config.lstm_hidden_dim,
            lstm_layers=config.lstm_layers,
            bidirectional=config.bidirectional,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout
        )
        print(f"✅ Created LSTMStage2Classifier: {input_dim}D×{config.sequence_length} → 3 classes")
        
    elif config.temporal_mode == 'relation':
        # Relation Network模式 - 需要从RelationFeatureFusion获取正确的维度
        from models.feature_extractors import RelationFeatureFusion
        temp_fusion = RelationFeatureFusion(
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context
        )
        
        person_feature_dim = temp_fusion.get_person_feature_dim()
        spatial_feature_dim = temp_fusion.get_spatial_feature_dim()
        # Validate person feature dim - relation mode requires per-person features (e.g., HoG)
        if person_feature_dim <= 0:
            raise ValueError(
                "Relation mode requires non-zero per-person features (enable HoG or adjust RelationFeatureFusion). "
                f"Got person_feature_dim={person_feature_dim}."
            )
        model = RelationStage2Classifier(
            person_feature_dim=person_feature_dim,  # 每个人的个体特征维度
            spatial_feature_dim=spatial_feature_dim,  # 空间关系特征维度
            hidden_dims=config.relation_hidden_dims,
            dropout=config.dropout,
            fusion_strategy=config.fusion_strategy
        )
        print(f"Created RelationStage2Classifier: Person({person_feature_dim}D)x2 + Spatial({spatial_feature_dim}D) -> 3 classes ({config.fusion_strategy})")
        
    else:
        raise ValueError(f"Unknown temporal_mode: {config.temporal_mode}")
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"Model parameters: {model_info['trainable_params']:,}")
    print(f"Model structure: {model_info['hidden_dims']}")
    
    return model


def create_stage2_loss(config: Stage2Config) -> Stage2Loss:
    """
    创建Stage2损失函数
    
    Args:
        config: Stage2配置对象
        
    Returns:
        Stage2Loss: 损失函数
    """
    # 如果config中没有class_weights，提供均匀权重作为兜底
    if not hasattr(config, 'class_weights') or config.class_weights is None:
        # 假设3类均匀权重
        config.class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
        print("⚠️ config.class_weights not found — using uniform fallback weights: {0:1.0,1:1.0,2:1.0}")

    criterion = Stage2Loss(
        class_weights=config.class_weights,
        mpca_weight=config.mpca_weight,
        acc_weight=config.acc_weight
    )

    print(f"✅ Created Stage2Loss: weights={config.class_weights}")
    return criterion


def create_optimizer(model: nn.Module, config: Stage2Config) -> optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        config: 配置对象
        
    Returns:
        torch.optim.Optimizer: 优化器
    """
    if config.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    print(f"✅ Created optimizer: {config.optimizer}, lr={config.learning_rate}")
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Stage2Config) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置对象
        
    Returns:
        Optional[_LRScheduler]: 学习率调度器 (可能为None)
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
            optimizer, mode='min', patience=5, factor=0.5
        )
    elif config.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    
    if scheduler:
        print(f"✅ Created scheduler: {config.scheduler}")
    else:
        print("✅ No scheduler used")
        
    return scheduler


def create_full_training_setup(config: Stage2Config, device: torch.device) -> Tuple[nn.Module, Stage2Loss, optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    """
    创建完整的训练设置
    
    Args:
        config: 配置对象
        device: 设备
        
    Returns:
        Tuple: (model, criterion, optimizer, scheduler)
    """
    print(f"\n🏗️ Creating training setup for {config.model_type}...")
    
    # 创建模型并移动到设备
    model = create_stage2_model(config).to(device)
    
    # 创建损失函数并移动到设备
    criterion = create_stage2_loss(config).to(device)
    
    # 创建优化器
    optimizer = create_optimizer(model, config)
    
    # 创建调度器
    scheduler = create_scheduler(optimizer, config)
    
    print(f"✅ Training setup completed on {device}")
    
    return model, criterion, optimizer, scheduler


class ModelCheckpointManager:
    """模型检查点管理器"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: Optional[optim.lr_scheduler._LRScheduler],
                       epoch: int, metrics: dict, filename: str, config: Stage2Config):
        """
        保存模型检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 调度器
            epoch: 当前epoch
            metrics: 评估指标
            filename: 文件名
            config: 配置
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': config.__dict__,
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }
        
        filepath = f"{self.save_dir}/{filename}.pth"
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filename}.pth")
    
    def load_checkpoint(self, filepath: str, model: nn.Module, 
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> dict:
        """
        加载模型检查点
        
        Args:
            filepath: 检查点文件路径
            model: 模型
            optimizer: 优化器 (可选)
            scheduler: 调度器 (可选)
            
        Returns:
            dict: 检查点信息
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📂 Checkpoint loaded from: {filepath}")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint


if __name__ == '__main__':
    # 测试模型工厂
    print("Testing Model Factory...")
    
    # 创建测试配置
    from configs.stage2_config import Stage2Config
    
    print("\n1. Testing Basic mode model creation...")
    config_basic = Stage2Config(
        temporal_mode='none',
        use_geometric=True,
        use_hog=True,
        use_scene_context=True,
        hidden_dims=[64, 32, 16],
        dropout=0.2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, criterion, optimizer, scheduler = create_full_training_setup(config_basic, device)
    
    print(f"\nModel type: {type(model).__name__}")
    print(f"Loss type: {type(criterion).__name__}")
    print(f"Optimizer type: {type(optimizer).__name__}")
    print(f"Scheduler type: {type(scheduler).__name__ if scheduler else 'None'}")
    
    # 测试前向传播
    print(f"\n2. Testing forward pass...")
    batch_size = 4
    input_dim = config_basic.get_input_dim()
    test_input = torch.randn(batch_size, input_dim).to(device)
    test_targets = torch.randint(0, 3, (batch_size,)).to(device)
    
    with torch.no_grad():
        logits = model(test_input)
        loss, loss_dict = criterion(logits, test_targets)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # 测试检查点管理器
    print(f"\n3. Testing checkpoint manager...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = ModelCheckpointManager(temp_dir)
        
        # 保存检查点
        test_metrics = {'val_accuracy': 0.85, 'val_mpca': 0.82}
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, 
            epoch=10, metrics=test_metrics, 
            filename='test_checkpoint', config=config_basic
        )
        
        # 加载检查点
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        loaded_checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        
        print(f"Loaded checkpoint epoch: {loaded_checkpoint.get('epoch')}")
    
    print("\n✅ Model factory test completed!")