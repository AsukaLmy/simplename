#!/usr/bin/env python3
"""
ResNet-based Stage2 Behavior Classification Models
Implementation of Relation Network with ResNet backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np

from models.resnet_feature_extractors import ResNetBackbone


class ResNetRelationStage2Classifier(nn.Module):
    """
    ResNet-based Relation Network for Stage2 behavior classification
    Architecture: ResNet visual features + spatial features -> relation reasoning -> classification
    """
    
    def __init__(self, person_feature_dim: int, spatial_feature_dim: int,
                 hidden_dims: List[int] = [512, 256, 128], 
                 dropout: float = 0.3, fusion_strategy: str = 'concat',
                 backbone_name: str = 'resnet18', pretrained: bool = True,
                 freeze_backbone: bool = False, crop_size: int = 224):
        """
        Args:
            person_feature_dim: 每个人的视觉特征维度
            spatial_feature_dim: 空间关系特征维度  
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比例
            fusion_strategy: 特征融合策略 ('concat', 'add', 'bilinear')
        """
        super().__init__()
        
        self.person_feature_dim = person_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.fusion_strategy = fusion_strategy

        # Create ResNet backbone inside the model so it can be fine-tuned / frozen by optimizer
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            feature_dim=person_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            input_size=crop_size
        )
        
        # Person-level feature processing
        self.person_encoder = nn.Sequential(
            nn.Linear(person_feature_dim, hidden_dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0] // 2),
            nn.ReLU(inplace=True)
        )
        
        # Spatial feature processing
        if spatial_feature_dim > 0:
            self.spatial_encoder = nn.Sequential(
                nn.Linear(spatial_feature_dim, hidden_dims[0] // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout // 2)
            )
        else:
            self.spatial_encoder = None
        
        # Relation reasoning module
        if fusion_strategy == 'concat':
            # Concatenate person features + spatial features
            relation_input_dim = (hidden_dims[0] // 2) * 2  # Two persons
            if spatial_feature_dim > 0:
                relation_input_dim += hidden_dims[0] // 4  # Spatial features
                
        elif fusion_strategy == 'bilinear':
            # Bilinear fusion between two persons
            self.bilinear = nn.Bilinear(hidden_dims[0] // 2, hidden_dims[0] // 2, hidden_dims[0])
            relation_input_dim = hidden_dims[0]
            if spatial_feature_dim > 0:
                relation_input_dim += hidden_dims[0] // 4
                
        elif fusion_strategy == 'add':
            # Element-wise addition (requires same dimension)
            assert person_feature_dim == spatial_feature_dim or spatial_feature_dim == 0
            relation_input_dim = hidden_dims[0] // 2
            if spatial_feature_dim > 0:
                relation_input_dim += hidden_dims[0] // 4
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Relation reasoning network
        relation_layers = []
        prev_dim = relation_input_dim
        
        for i, dim in enumerate(hidden_dims):
            relation_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout / 2),
            ])
            prev_dim = dim
        
        # Final classification layer
        relation_layers.extend([
            nn.Linear(prev_dim, 3),  # 3 classes: Walking Together, Standing Together, Sitting Together
        ])
        
        self.relation_network = nn.Sequential(*relation_layers)
        
        # Initialize weights
        self._init_weights()
        
        print(f"ResNet Relation Network created:")
        print(f"  Person features: {person_feature_dim}D -> {hidden_dims[0]//2}D")
        print(f"  Spatial features: {spatial_feature_dim}D -> {hidden_dims[0]//4 if spatial_feature_dim > 0 else 0}D")
        print(f"  Fusion strategy: {fusion_strategy}")
        print(f"  Relation input: {relation_input_dim}D")
        print(f"  Hidden layers: {hidden_dims}")
        print(f"  Output: 3 classes")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, person_A_features: torch.Tensor, person_B_features: torch.Tensor, 
                spatial_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            person_A_features: [batch_size, person_feature_dim] 人A特征
            person_B_features: [batch_size, person_feature_dim] 人B特征
            spatial_features: [batch_size, spatial_feature_dim] 空间关系特征
            
        Returns:
            torch.Tensor: [batch_size, 3] 分类logits
        """
        # If inputs are image tensors, run through backbone to get visual features
        # Expect person_A_features/person_B_features to be either [B, D] (precomputed) or [B,3,H,W] images
        if person_A_features.dim() == 4 and person_B_features.dim() == 4:
            # move image tensors to backbone device
            device = next(self.backbone.parameters()).device
            person_A_images = person_A_features.to(device)
            person_B_images = person_B_features.to(device)
            with torch.no_grad() if self.backbone.freeze_backbone else torch.enable_grad():
                person_A_feat = self.backbone(person_A_images)  # [B, person_feature_dim]
                person_B_feat = self.backbone(person_B_images)  # [B, person_feature_dim]
        else:
            person_A_feat = person_A_features
            person_B_feat = person_B_features

        # Encode individual person features
        person_A_encoded = self.person_encoder(person_A_feat)  # [B, hidden_dims[0]//2]
        person_B_encoded = self.person_encoder(person_B_feat)  # [B, hidden_dims[0]//2]
        
        # Encode spatial features
        if self.spatial_encoder is not None and spatial_features.numel() > 0:
            spatial_encoded = self.spatial_encoder(spatial_features)  # [B, hidden_dims[0]//4]
        else:
            spatial_encoded = None
        
        # Relation fusion
        if self.fusion_strategy == 'concat':
            # Concatenate all features
            relation_input = torch.cat([person_A_encoded, person_B_encoded], dim=1)
            if spatial_encoded is not None:
                relation_input = torch.cat([relation_input, spatial_encoded], dim=1)
                
        elif self.fusion_strategy == 'bilinear':
            # Bilinear interaction between two persons
            relation_input = self.bilinear(person_A_encoded, person_B_encoded)
            if spatial_encoded is not None:
                relation_input = torch.cat([relation_input, spatial_encoded], dim=1)
                
        elif self.fusion_strategy == 'add':
            # Element-wise addition
            relation_input = person_A_encoded + person_B_encoded
            if spatial_encoded is not None:
                relation_input = torch.cat([relation_input, spatial_encoded], dim=1)
        
        # Relation reasoning and classification
        logits = self.relation_network(relation_input)  # [B, 3]
        
        return logits
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ResNetRelationStage2Classifier',
            'person_feature_dim': self.person_feature_dim,
            'spatial_feature_dim': self.spatial_feature_dim,
            'hidden_dims': self.hidden_dims,
            'fusion_strategy': self.fusion_strategy,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dropout': self.dropout
        }


class ResNetStage2Loss(nn.Module):
    """
    Simplified Stage2 Loss function for ResNet-based models
    Uses only CrossEntropy loss
    """
    
    def __init__(self, class_weights: Optional[Dict] = None):
        """
        Args:
            class_weights: 类别权重字典
        """
        super().__init__()
        
        # 处理类别权重
        if class_weights is None:
            class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
        
        if isinstance(class_weights, dict):
            # 转换为tensor
            max_class = max(class_weights.keys())
            class_weights_tensor = torch.ones(max_class + 1, dtype=torch.float32)
            for class_id, weight in class_weights.items():
                class_weights_tensor[class_id] = weight
        else:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        self.register_buffer('class_weights', class_weights_tensor)
        
        print(f"ResNet Stage2 Loss created: CrossEntropy only, weights={class_weights}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        计算简化的交叉熵损失
        
        Args:
            predictions: [batch_size, 3] 预测logits
            targets: [batch_size] 目标标签
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # 加权交叉熵损失
        ce_loss = F.cross_entropy(
            predictions, targets, weight=self.class_weights, reduction='mean'
        )
        
        # 计算准确率用于记录
        with torch.no_grad():
            predicted_classes = torch.argmax(predictions, dim=1)
            overall_acc = (predicted_classes == targets).float().mean()
        
        loss_dict = {
            'total_loss': ce_loss.item(),
            'ce_loss': ce_loss.item(),
            'mpca_loss': 0.0,  # 保持接口兼容性
            'overall_acc': overall_acc.item()
        }
        
        return ce_loss, loss_dict


if __name__ == '__main__':
    # 测试ResNet Relation Network
    print("Testing ResNet Relation Stage2 Classifier...")
    
    # 测试参数
    batch_size = 4
    person_feature_dim = 256  # ResNet特征维度
    spatial_feature_dim = 8   # 几何(7) + 场景(1)
    
    # 创建模型
    model = ResNetRelationStage2Classifier(
        person_feature_dim=person_feature_dim,
        spatial_feature_dim=spatial_feature_dim,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        fusion_strategy='concat'
    )
    
    # 创建测试数据
    person_A_features = torch.randn(batch_size, person_feature_dim)
    person_B_features = torch.randn(batch_size, person_feature_dim)
    spatial_features = torch.randn(batch_size, spatial_feature_dim)
    targets = torch.randint(0, 3, (batch_size,))
    
    print(f"\nInput shapes:")
    print(f"  Person A: {person_A_features.shape}")
    print(f"  Person B: {person_B_features.shape}")
    print(f"  Spatial: {spatial_features.shape}")
    
    # 前向传播
    with torch.no_grad():
        logits = model(person_A_features, person_B_features, spatial_features)
        predictions = torch.argmax(logits, dim=1)
    
    print(f"\nOutput:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Predictions: {predictions.tolist()}")
    print(f"  Targets: {targets.tolist()}")
    
    # 测试损失函数
    criterion = ResNetStage2Loss(
        class_weights={0: 1.0, 1: 1.4, 2: 6.1},
        mpca_weight=0.1,
        acc_weight=0.05
    )
    
    loss, loss_dict = criterion(logits, targets)
    print(f"\nLoss:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss details: {loss_dict}")
    
    # 模型信息
    model_info = model.get_model_info()
    print(f"\nModel info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ ResNet Relation Stage2 Classifier test completed!")