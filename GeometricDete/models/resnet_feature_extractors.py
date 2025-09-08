#!/usr/bin/env python3
"""
ResNet-based Feature Extractors for Stage2 Behavior Classification
Uses pretrained ResNet as backbone for visual feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict
import os
from PIL import Image

# 导入几何特征提取函数
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from geometric_features import extract_geometric_features
except ImportError as e:
    print(f"Warning: Could not import geometric_features: {e}")


class ResNetBackbone(nn.Module):
    """
    预训练ResNet backbone for visual feature extraction
    """
    
    def __init__(self, backbone_name: str = 'resnet18', feature_dim: int = 256, 
                 pretrained: bool = True, freeze_backbone: bool = False, input_size: int = 224):
        """
        Args:
            backbone_name: ResNet架构名称 ('resnet18', 'resnet34', 'resnet50')
            feature_dim: 输出特征维度
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone参数
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # 创建backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # 移除最后的全连接层，只保留特征提取部分
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 去除fc层
        
        # 添加自适应池化确保固定输出尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征映射层
        self.feature_mapper = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 冻结backbone参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 图像预处理
        self.input_size = input_size
        self.preprocess = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),  # ResNet输入尺寸（可配置）
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
        
        print(f"Created {backbone_name} backbone: {backbone_dim}D -> {feature_dim}D, "
              f"pretrained={pretrained}, frozen={freeze_backbone}")
    
    def freeze_early_layers(self, freeze_layers: int = 3):
        """
        冻结ResNet前几个层（部分冻结策略）
        
        Args:
            freeze_layers: 冻结前几个残差块 (1-4)
        """
        children = list(self.backbone.children())
        layers_to_freeze = children[:freeze_layers+3]  # conv1, bn1, relu, maxpool + 前N个残差块
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        
        print(f"Frozen first {freeze_layers} residual blocks of {self.backbone_name}")
    
    def unfreeze_last_layers(self, unfreeze_layers: int = 1):
        """
        解冻ResNet后几个层（渐进式解冻）
        
        Args:
            unfreeze_layers: 解冻后几个残差块
        """
        children = list(self.backbone.children())
        layers_to_unfreeze = children[-unfreeze_layers:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"Unfrozen last {unfreeze_layers} residual blocks of {self.backbone_name}")
    
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        提取视觉特征
        
        Args:
            image_tensor: [B, C, H, W] 预处理后的图像tensor
            
        Returns:
            torch.Tensor: [B, feature_dim] 特征向量
        """
        # 通过backbone提取特征
        features = self.backbone(image_tensor)  # [B, backbone_dim, H', W']
        
        # 自适应池化到固定尺寸
        features = self.adaptive_pool(features)  # [B, backbone_dim, 1, 1]
        features = features.view(features.size(0), -1)  # [B, backbone_dim]
        
        # 特征映射
        features = self.feature_mapper(features)  # [B, feature_dim]
        
        return features
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        预处理单张图像
        
        Args:
            image: PIL Image对象
            
        Returns:
            torch.Tensor: [1, 3, 224, 224] 预处理后的图像tensor
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        tensor = self.preprocess(image).unsqueeze(0)  # 添加batch维度
        return tensor


class PersonCropExtractor(nn.Module):
    """
    Person region cropping and feature extraction
    """
    
    def __init__(self, backbone: ResNetBackbone, crop_size: int = 112, 
                 padding_ratio: float = 0.2):
        """
        Args:
            backbone: ResNet backbone
            crop_size: 裁剪区域目标尺寸
            padding_ratio: 边界框padding比例
        """
        super().__init__()
        self.backbone = backbone
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        
        # 调整预处理以适应较小的裁剪区域
        self.crop_preprocess = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),  # 适应较小区域
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def crop_person_region(self, image: Image.Image, bbox: torch.Tensor, 
                          image_width: int = 3760, image_height: int = 480) -> Image.Image:
        """
        裁剪人体区域
        
        Args:
            image: PIL Image对象
            bbox: [4] 边界框 [x, y, w, h]
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            Image.Image: 裁剪的人体区域
        """
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        
        x, y, w, h = bbox
        
        # 添加padding
        padding_w = w * self.padding_ratio
        padding_h = h * self.padding_ratio
        
        # 计算裁剪区域
        x1 = max(0, int(x - padding_w))
        y1 = max(0, int(y - padding_h))
        x2 = min(image_width, int(x + w + padding_w))
        y2 = min(image_height, int(y + h + padding_h))
        
        # 确保区域有效
        if x2 <= x1 or y2 <= y1:
            # 使用最小有效区域
            center_x, center_y = int(x + w/2), int(y + h/2)
            min_size = max(32, int(max(w, h)))
            x1 = max(0, center_x - min_size//2)
            y1 = max(0, center_y - min_size//2)
            x2 = min(image_width, x1 + min_size)
            y2 = min(image_height, y1 + min_size)
        
        # 裁剪区域
        person_region = image.crop((x1, y1, x2, y2))
        
        return person_region
    
    def forward(self, image: Image.Image, bbox: torch.Tensor, 
                image_width: int = 3760, image_height: int = 480) -> torch.Tensor:
        """
        提取单个人的视觉特征
        
        Args:
            image: PIL Image对象
            bbox: [4] 边界框
            image_width: 图像宽度  
            image_height: 图像高度
            
        Returns:
            torch.Tensor: [feature_dim] 人体特征向量
        """
        try:
            # 裁剪人体区域
            person_region = self.crop_person_region(image, bbox, image_width, image_height)
            
            # 预处理
            if person_region.mode != 'RGB':
                person_region = person_region.convert('RGB')
            
            image_tensor = self.crop_preprocess(person_region).unsqueeze(0)  # [1, 3, H, W]

            # Ensure backbone and input are on same device
            try:
                device = next(self.backbone.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            image_tensor = image_tensor.to(device)

            # 提取特征
            with torch.no_grad():
                features = self.backbone(image_tensor)  # [1, feature_dim]
            
            return features.squeeze(0)  # [feature_dim]
            
        except Exception as e:
            print(f"Warning: Person feature extraction failed: {e}")
            # 返回零向量作为fallback
            return torch.zeros(self.backbone.feature_dim, dtype=torch.float32)


class ResNetRelationFeatureFusion(nn.Module):
    """
    ResNet-based Relation Network特征融合器
    使用预训练ResNet提取人体视觉特征，结合几何和场景特征
    """
    
    def __init__(self, backbone_name: str = 'resnet18', visual_feature_dim: int = 256,
                 use_geometric: bool = True, use_scene_context: bool = True,
                 pretrained: bool = True, freeze_backbone: bool = False, crop_size: int = 224):
        """
        Args:
            backbone_name: ResNet架构名称
            visual_feature_dim: 视觉特征维度
            use_geometric: 是否使用几何特征
            use_scene_context: 是否使用场景上下文
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.visual_feature_dim = visual_feature_dim
        self.use_geometric = use_geometric
        self.use_scene_context = use_scene_context
        
        # 创建ResNet backbone (input size aligned with crop_size)
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            feature_dim=visual_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            input_size=crop_size
        )

        # 创建人体特征提取器
        self.person_extractor = PersonCropExtractor(self.backbone, crop_size=crop_size)
        
        # 场景上下文提取器
        if self.use_scene_context:
            self.scene_extractor = SceneContextExtractor()
        
        # 计算特征维度
        self.person_feature_dim = self._calculate_person_feature_dim()
        self.spatial_feature_dim = self._calculate_spatial_feature_dim()
        
        print(f"ResNet Relation Feature Fusion created:")
        print(f"  Person features: {self.person_feature_dim}D (ResNet {backbone_name})")
        print(f"  Spatial features: {self.spatial_feature_dim}D")
    
    def _calculate_person_feature_dim(self) -> int:
        """计算每个人的特征维度"""
        return self.visual_feature_dim  # ResNet输出维度
    
    def _calculate_spatial_feature_dim(self) -> int:
        """计算空间关系特征维度"""
        dim = 0
        if self.use_geometric:
            dim += 7  # 几何关系特征
        if self.use_scene_context:
            dim += 1  # 场景上下文
        return dim
    
    def forward(self, person_A_box: torch.Tensor, person_B_box: torch.Tensor,
                image: Optional[Image.Image] = None, 
                all_boxes: Optional[list] = None,
                image_width: int = 3760, image_height: int = 480) -> Dict[str, torch.Tensor]:
        """
        提取relation network所需的分离特征
        
        Args:
            person_A_box: [4] 人A边界框
            person_B_box: [4] 人B边界框
            image: PIL Image对象
            all_boxes: 所有人的边界框列表
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            Dict: {
                'person_A_features': [person_feature_dim] 人A特征
                'person_B_features': [person_feature_dim] 人B特征  
                'spatial_features': [spatial_feature_dim] 空间关系特征
            }
        """
        # 提取个体视觉特征
        if image is not None:
            # 提取人A特征
            person_A_features = self.person_extractor(image, person_A_box, image_width, image_height)
            
            # 提取人B特征
            person_B_features = self.person_extractor(image, person_B_box, image_width, image_height)
        else:
            # 如果没有图像，使用零向量
            print("Warning: No image provided, using zero visual features")
            person_A_features = torch.zeros(self.visual_feature_dim, dtype=torch.float32)
            person_B_features = torch.zeros(self.visual_feature_dim, dtype=torch.float32)
        
        # 提取空间关系特征
        spatial_features = []
        
        # 几何特征
        if self.use_geometric:
            try:
                geometric_features = extract_geometric_features(
                    person_A_box, person_B_box, image_width, image_height
                )
                if isinstance(geometric_features, torch.Tensor):
                    geometric_features = geometric_features.cpu().numpy()
                spatial_features.append(torch.tensor(geometric_features, dtype=torch.float32))
            except Exception as e:
                print(f"Warning: Geometric feature extraction failed: {e}")
                spatial_features.append(torch.zeros(7, dtype=torch.float32))
        
        # 场景上下文特征
        if self.use_scene_context and all_boxes is not None:
            try:
                scene_features = self.scene_extractor(all_boxes)
                spatial_features.append(scene_features)
            except Exception as e:
                print(f"Warning: Scene context extraction failed: {e}")
                spatial_features.append(torch.zeros(1, dtype=torch.float32))
        
        # 组合空间特征并确保固定维度
        if self.spatial_feature_dim > 0:
            if spatial_features:
                spatial_tensor = torch.cat(spatial_features, dim=0)
                # pad or trim to exact spatial_feature_dim
                if spatial_tensor.numel() < self.spatial_feature_dim:
                    pad = torch.zeros(self.spatial_feature_dim - spatial_tensor.numel(), dtype=torch.float32)
                    spatial_tensor = torch.cat([spatial_tensor, pad], dim=0)
                elif spatial_tensor.numel() > self.spatial_feature_dim:
                    spatial_tensor = spatial_tensor[:self.spatial_feature_dim]
            else:
                spatial_tensor = torch.zeros(self.spatial_feature_dim, dtype=torch.float32)
        else:
            spatial_tensor = torch.zeros(0, dtype=torch.float32)
        
        return {
            'person_A_features': person_A_features,
            'person_B_features': person_B_features,
            'spatial_features': spatial_tensor
        }
    
    def get_person_feature_dim(self) -> int:
        """获取单个人的特征维度"""
        return self.person_feature_dim
    
    def get_spatial_feature_dim(self) -> int:
        """获取空间关系特征维度"""
        return self.spatial_feature_dim
    
    def get_feature_info(self) -> dict:
        """获取特征信息"""
        return {
            'backbone': self.backbone_name,
            'visual_feature_dim': self.visual_feature_dim,
            'person_feature_dim': self.person_feature_dim,
            'spatial_feature_dim': self.spatial_feature_dim,
            'geometric': {'enabled': self.use_geometric, 'dim': 7 if self.use_geometric else 0},
            'scene_context': {'enabled': self.use_scene_context, 'dim': 1 if self.use_scene_context else 0}
        }


class SceneContextExtractor(nn.Module):
    """
    场景上下文特征提取器 - 复用原有实现
    """
    
    def __init__(self):
        super().__init__()
        self.feature_dim = 1
        
    def forward(self, all_boxes: list) -> torch.Tensor:
        """
        提取场景上下文特征
        
        Args:
            all_boxes: 当前帧中所有人的边界框列表
            
        Returns:
            torch.Tensor: [1] 场景上下文特征 (场景密度)
        """
        if not all_boxes:
            return torch.tensor([1.0], dtype=torch.float32)  # 默认稀疏场景
        
        # 场景密度计算
        num_people = len(all_boxes)
        scene_density = min(num_people / 10.0, 1.0)  # 归一化到[0,1]
        return torch.tensor([scene_density], dtype=torch.float32)


if __name__ == '__main__':
    # 测试ResNet特征提取器
    print("Testing ResNet Feature Extractors...")
    
    # 测试参数
    test_image = Image.new('RGB', (3760, 480), (128, 128, 128))
    person_A_box = torch.tensor([1000, 200, 100, 200], dtype=torch.float32)
    person_B_box = torch.tensor([1200, 180, 120, 220], dtype=torch.float32)
    all_boxes = [[1000, 200, 100, 200], [1200, 180, 120, 220], [800, 150, 80, 180]]
    
    print("\n1. Testing ResNet18 backbone...")
    feature_fusion = ResNetRelationFeatureFusion(
        backbone_name='resnet18',
        visual_feature_dim=256,
        pretrained=True,
        freeze_backbone=False
    )
    
    # 提取特征
    features = feature_fusion(person_A_box, person_B_box, test_image, all_boxes)
    
    print(f"Person A features shape: {features['person_A_features'].shape}")
    print(f"Person B features shape: {features['person_B_features'].shape}")
    print(f"Spatial features shape: {features['spatial_features'].shape}")
    
    # 特征统计
    person_A_stats = features['person_A_features']
    print(f"Person A stats: min={person_A_stats.min():.4f}, max={person_A_stats.max():.4f}, mean={person_A_stats.mean():.4f}")
    
    print(f"\nFeature info: {feature_fusion.get_feature_info()}")
    
    print("\n✅ ResNet feature extractors test completed!")