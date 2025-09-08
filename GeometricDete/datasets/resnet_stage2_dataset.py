#!/usr/bin/env python3
"""
ResNet-based Stage2 Dataset
Dataset implementation for ResNet backbone with Relation Network
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import Counter

# 导入基础组件
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.resnet_feature_extractors import ResNetRelationFeatureFusion
from datasets.stage2_dataset import Stage2LabelMapper  # 复用标签映射器
from geometric_features import extract_geometric_features


class ResNetStage2Dataset(Dataset):
    """
    ResNet-based Stage2 dataset for behavior classification
    Compatible with Relation Network architecture using ResNet backbone
    """
    
    def __init__(self, data_path: str, split: str = 'train',
                 backbone_name: str = 'resnet18', visual_feature_dim: int = 256,
                 use_geometric: bool = True, use_scene_context: bool = True,
                 pretrained: bool = True, freeze_backbone: bool = False,
                 frame_interval: int = 1, use_oversampling: bool = False,
                 crop_size: int = 112):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            backbone_name: ResNet架构名称
            visual_feature_dim: 视觉特征维度
            use_geometric: 是否使用几何特征
            use_scene_context: 是否使用场景上下文
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone
            frame_interval: 帧采样间隔
            use_oversampling: 是否使用过采样
        """
        self.data_path = data_path
        self.split = split
        self.backbone_name = backbone_name
        self.visual_feature_dim = visual_feature_dim
        self.use_geometric = use_geometric
        self.use_scene_context = use_scene_context
        self.frame_interval = frame_interval
        self.use_oversampling = use_oversampling
        
        # 创建标签映射器
        self.label_mapper = Stage2LabelMapper()
        
        # 创建特征融合器
        self.feature_fusion = ResNetRelationFeatureFusion(
            backbone_name=backbone_name,
            visual_feature_dim=visual_feature_dim,
            use_geometric=use_geometric,
            use_scene_context=use_scene_context,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            crop_size=crop_size
        )

        # store crop size
        self.crop_size = crop_size

        # 固定的场景划分 (复用原有划分)
        self.trainset_split = [
        'bytes-cafe-2019-02-07_0', 
        'clark-center-2019-02-28_0', 
        'cubberly-auditorium-2019-04-22_0',
        'forbes-cafe-2019-01-22_0', 
        'gates-159-group-meeting-2019-04-03_0',
        'gates-to-clark-2019-02-28_1',
        'gates-ai-lab-2019-02-08_0',
        'hewlett-packard-intersection-2019-01-24_0', 
        'huang-2-2019-01-25_0', 
        'huang-basement-2019-01-25_0',
        'huang-lane-2019-02-12_0', 
        'memorial-court-2019-03-16_0', 
        'meyer-green-2019-03-16_0',
        'nvidia-aud-2019-04-18_0', 
        'packard-poster-session-2019-03-20_2', 
        'stlc-111-2019-04-19_0', 
        'svl-meeting-gates-2-2019-04-08_0',
        'tressider-2019-04-26_2',
        'jordan-hall-2019-04-22_0', 
        ]
        
        self.valset_split = [
        'clark-center-2019-02-28_1',
        'gates-basement-elevators-2019-01-17_1', 
        'packard-poster-session-2019-03-20_1',
        'svl-meeting-gates-2-2019-04-08_1',
        'tressider-2019-03-16_1',
        ]
        
        self.testset_split = [
        'packard-poster-session-2019-03-20_0',
        'clark-center-intersection-2019-02-28_0', 
        
        'stlc-111-2019-04-19_0', 
        'tressider-2019-03-16_0',
        ]
        
        # 加载数据
        self._load_data()

        # Dataset returns cropped image tensors; backbone is inside model for finetuning
        # Debug prints (use instance attributes so they run during construction, not import)
        print(f"ResNet Stage2 Dataset created:")
        print(f"  Split: {self.split}")
        print(f"  Backbone: {self.backbone_name}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Visual features: {self.visual_feature_dim}D")
        print(f"  Frame interval: {self.frame_interval}")

    def _load_data(self):
        """加载JRDB格式的社交标签数据"""
        # 使用JRDB数据结构
        social_labels_dir = os.path.join(self.data_path, 'labels', 'labels_2d_activity_social_stitched')
        images_dir = os.path.join(self.data_path, 'images', 'image_stitched')
        
        if not os.path.exists(social_labels_dir):
            raise FileNotFoundError(f"Social labels directory not found: {social_labels_dir}")
        
        # 根据split选择场景
        if self.split == 'train':
            scene_splits = self.trainset_split
        elif self.split == 'val':
            scene_splits = self.valset_split
        elif self.split == 'test':
            scene_splits = self.testset_split
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        self.samples = []
        self.stage2_labels = []
        
        # 获取场景文件
        scene_files = [f for f in os.listdir(social_labels_dir) if f.endswith('.json')]
        scene_files.sort()
        
        # 筛选存在的场景文件
        selected_files = []
        for scene_name in scene_splits:
            scene_file = f"{scene_name}.json"
            if scene_file in scene_files:
                selected_files.append(scene_file)
            else:
                print(f"Warning: Scene file {scene_file} not found in dataset")
        
        print(f"Loading {len(selected_files)}/{len(scene_splits)} scenes for {self.split} split")
        
        # 加载数据
        sample_count = 0
        for scene_file in selected_files:
            scene_path = os.path.join(social_labels_dir, scene_file)
            scene_name = os.path.splitext(scene_file)[0]
            
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
                
                # 处理场景中的每一帧
                frame_names = list(scene_data.get('labels', {}).keys())
                frame_names.sort()
                
                # 应用帧间隔采样
                selected_frames = frame_names[::self.frame_interval]
                
                for image_name in selected_frames:
                    annotations = scene_data['labels'][image_name]
                    frame_id = f"{scene_name}_{self._extract_frame_id(image_name)}"
                    
                    # 构建图像路径
                    image_path = os.path.join(images_dir, scene_name, image_name)
                    
                    # 收集该帧的所有人员信息
                    person_dict = {}
                    all_boxes = []
                    
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if person_id.startswith('pedestrian:'):
                            pid = int(person_id.split(':')[1])
                            box = ann.get('box', [0, 0, 100, 100])
                            
                            # 数据验证：检查边界框有效性
                            if self._is_valid_box(box):
                                all_boxes.append(box)
                                person_dict[pid] = {
                                    'box': box,
                                    'interactions': ann.get('H-interaction', [])
                                }
                    
                    # 提取正样本（有交互的人员对）
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if not person_id.startswith('pedestrian:'):
                            continue
                        
                        person_A_id = int(person_id.split(':')[1])
                        if person_A_id not in person_dict:
                            continue
                        
                        person_A_box = person_dict[person_A_id]['box']
                        
                        # 处理该人员的所有交互
                        for interaction in ann.get('H-interaction', []):
                            pair_id = interaction.get('pair', '')
                            if pair_id.startswith('pedestrian:'):
                                person_B_id = int(pair_id.split(':')[1])
                                
                                if person_B_id in person_dict:
                                    # 避免重复交互对：只保留ID较小者作为person_A
                                    if person_A_id < person_B_id:
                                        person_B_box = person_dict[person_B_id]['box']
                                        interaction_labels = interaction.get('inter_labels', {})
                                        
                                        # 检查是否为有效的Stage2交互
                                        if isinstance(interaction_labels, dict) and len(interaction_labels) > 0:
                                            interaction_type = list(interaction_labels.keys())[0]
                                            
                                            # 映射到Stage2标签
                                            stage2_label = self.label_mapper.map_label(interaction_type)
                                            
                                            if stage2_label is not None:
                                                sample = {
                                                    'image_path': image_path if os.path.exists(image_path) else None,
                                                    'person_A_box': torch.tensor(person_A_box, dtype=torch.float32),
                                                    'person_B_box': torch.tensor(person_B_box, dtype=torch.float32),
                                                    'stage2_label': stage2_label,
                                                    'scene_name': scene_name,
                                                    'frame_id': frame_id,
                                                    'all_boxes': all_boxes,
                                                    'original_interaction': interaction_type
                                                }
                                                
                                                self.samples.append(sample)
                                                self.stage2_labels.append(stage2_label)
                                                sample_count += 1
                    
                    # 定期打印进度
                    if sample_count % 1000 == 0 and sample_count > 0:
                        print(f"  Processed {sample_count} samples...")
                        
            except Exception as e:
                print(f"Warning: Error loading scene {scene_file}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples from {len(selected_files)} scenes")
        if self.frame_interval > 1:
            print(f"Applied frame interval {self.frame_interval} (reduced samples by ~{((self.frame_interval-1)/self.frame_interval)*100:.0f}%)")
        
        # 打印类别分布
        if self.stage2_labels:
            label_counts = Counter(self.stage2_labels)
            print(f"Label distribution: {dict(label_counts)}")
    
    def _extract_frame_id(self, image_name: str) -> str:
        """从图像名提取帧ID"""
        # JRDB format: "000000.jpg" -> "000000"
        return os.path.splitext(image_name)[0]
    
    def _is_valid_box(self, box: List[float]) -> bool:
        """验证边界框的有效性"""
        if len(box) != 4:
            return False
        x, y, w, h = box
        # 检查边界框的合理性
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return False
        if w > 5000 or h > 5000:  # 异常大的边界框
            return False
        return True
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        
        Returns:
            Dict包含:
            - person_A_features: [visual_feature_dim] A的视觉特征
            - person_B_features: [visual_feature_dim] B的视觉特征  
            - spatial_features: [spatial_feature_dim] 空间关系特征
            - stage2_label: int 行为标签
        """
        sample = self.samples[idx]

        # 获取图像
        image = None
        image_path = sample['image_path']
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")

        # Return cropped image tensors for person A and B (so model performs backbone forward)
        # PersonCropExtractor has an internal preprocess that resizes to crop_size
        if image is not None:
            person_A_img_tensor = self.feature_fusion.person_extractor.crop_person_region(image, sample['person_A_box'], image_width=3760, image_height=480)
            person_B_img_tensor = self.feature_fusion.person_extractor.crop_person_region(image, sample['person_B_box'], image_width=3760, image_height=480)

            # preprocess to tensor
            person_A_tensor = self.feature_fusion.person_extractor.crop_preprocess(person_A_img_tensor).unsqueeze(0)  # [1,3,H,W]
            person_B_tensor = self.feature_fusion.person_extractor.crop_preprocess(person_B_img_tensor).unsqueeze(0)
            # squeeze batch dim for consistency with model expected [B,3,H,W] per batch later
            # we'll return tensors without batch dimension; DataLoader collate will batch them
        else:
            # fallback zero images
            person_A_tensor = torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32)
            person_B_tensor = torch.zeros(3, self.crop_size, self.crop_size, dtype=torch.float32)

        # spatial features: compute using feature_fusion helpers (without computing visual features)
        spatial_feats = []
        if self.use_geometric:
            try:
                geom = extract_geometric_features(sample['person_A_box'], sample['person_B_box'], 3760, 480)
                if isinstance(geom, torch.Tensor):
                    geom = geom.cpu().numpy()
                spatial_feats.append(torch.tensor(geom, dtype=torch.float32))
            except Exception:
                spatial_feats.append(torch.zeros(7, dtype=torch.float32))
        if self.use_scene_context:
            try:
                scene = self.feature_fusion.scene_extractor(sample.get('all_boxes', []))
                spatial_feats.append(scene)
            except Exception:
                spatial_feats.append(torch.zeros(1, dtype=torch.float32))

        if spatial_feats:
            spatial_tensor = torch.cat(spatial_feats, dim=0)
        else:
            spatial_tensor = torch.zeros(self.feature_fusion.get_spatial_feature_dim(), dtype=torch.float32)

        return {
            'person_A_features': person_A_tensor,  # [3,H,W]
            'person_B_features': person_B_tensor,
            'spatial_features': spatial_tensor,    # [spatial_dim]
            'stage2_label': torch.tensor(sample['stage2_label'], dtype=torch.long)
        }

    def _get_cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.split}_resnet_features.pth")

    def _ensure_feature_cache(self):
        """If cache exists, load cached feature tensors into samples; otherwise compute and save cache."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location='cpu')
                if not isinstance(data, dict):
                    raise ValueError("Invalid cache format")
                # load cached features into samples
                for i, feat in enumerate(data.get('features', [])):
                    if i < len(self.samples):
                        self.samples[i]['cached_person_A'] = feat['person_A']
                        self.samples[i]['cached_person_B'] = feat['person_B']
                        self.samples[i]['cached_spatial'] = feat['spatial']
                print(f"Loaded feature cache: {cache_path}")
                return
            except Exception as e:
                print(f"Warning: Failed to load feature cache: {e}, will recompute")

        # compute and cache
        print(f"Precomputing visual+spatial features for {len(self.samples)} samples (this may take a while)...")
        features_list = []
        for i, sample in enumerate(self.samples):
            image = None
            image_path = sample['image_path']
            if image_path and os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception:
                    image = None

            feats = self.feature_fusion(
                person_A_box=sample['person_A_box'],
                person_B_box=sample['person_B_box'],
                image=image,
                all_boxes=sample.get('all_boxes', [])
            )

            person_A_feat = feats['person_A_features'].detach().cpu()
            person_B_feat = feats['person_B_features'].detach().cpu()
            spatial_feat = feats['spatial_features'].detach().cpu()

            # ensure fixed spatial dim
            if spatial_feat.numel() != self.feature_fusion.get_spatial_feature_dim() and self.feature_fusion.get_spatial_feature_dim() > 0:
                desired = self.feature_fusion.get_spatial_feature_dim()
                cur = spatial_feat.numel()
                if cur < desired:
                    pad = torch.zeros(desired - cur, dtype=torch.float32)
                    spatial_feat = torch.cat([spatial_feat, pad], dim=0)
                else:
                    spatial_feat = spatial_feat[:desired]

            self.samples[i]['cached_person_A'] = person_A_feat
            self.samples[i]['cached_person_B'] = person_B_feat
            self.samples[i]['cached_spatial'] = spatial_feat

            features_list.append({
                'person_A': person_A_feat,
                'person_B': person_B_feat,
                'spatial': spatial_feat
            })

            if (i + 1) % 500 == 0:
                print(f"  Precomputed {i+1}/{len(self.samples)} samples")

        # save cache
        try:
            torch.save({'features': features_list}, cache_path)
            print(f"Saved feature cache to: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save feature cache: {e}")
    
    def get_labels(self) -> List[int]:
        """获取所有样本的标签"""
        return self.stage2_labels.copy()
    
    def get_class_distribution(self) -> Dict:
        """获取类别分布信息"""
        if not self.stage2_labels:
            return {"message": "No labels available"}
        
        label_counts = Counter(self.stage2_labels)
        total = len(self.stage2_labels)
        
        class_names = self.label_mapper.class_names
        
        return {
            'total': total,
            'class_counts': dict(label_counts),
            'class_names': class_names,
            'class_weights': {k: total / (len(label_counts) * v) for k, v in label_counts.items()}
        }
    
    def get_feature_info(self) -> Dict:
        """获取特征信息"""
        return self.feature_fusion.get_feature_info()


# 数据加载器创建函数
def create_resnet_stage2_data_loaders(config) -> Tuple:
    """
    创建ResNet Stage2数据加载器
    
    Args:
        config: ResNetStage2Config配置对象
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from collections import Counter
    
    print(f"Creating ResNet Stage2 data loaders...")
    print(f"  Backbone: {config.backbone_name}")
    print(f"  Visual features: {config.visual_feature_dim}D")
    print(f"  Frame interval: {config.frame_interval}")
    
    # 创建数据集
    train_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='train',
        backbone_name=config.backbone_name,
        visual_feature_dim=config.visual_feature_dim,
        use_geometric=config.use_geometric,
        use_scene_context=config.use_scene_context,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        frame_interval=config.frame_interval,
        use_oversampling=True
    )
    
    val_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='val',
        backbone_name=config.backbone_name,
        visual_feature_dim=config.visual_feature_dim,
        use_geometric=config.use_geometric,
        use_scene_context=config.use_scene_context,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        frame_interval=config.frame_interval,
        use_oversampling=False
    )
    
    test_dataset = ResNetStage2Dataset(
        data_path=config.data_path,
        split='test',
        backbone_name=config.backbone_name,
        visual_feature_dim=config.visual_feature_dim,
        use_geometric=config.use_geometric,
        use_scene_context=config.use_scene_context,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        frame_interval=config.frame_interval,
        use_oversampling=False
    )
    
    # 创建训练集采样器（用于类别平衡）
    train_sampler = None
    if config.use_oversampling:
        try:
            labels = train_dataset.get_labels()
            if labels and len(labels) > 0:
                counts = Counter(labels)
                total = len(labels)
                
                # 计算类别权重并更新config
                class_weights = {int(c): total / (len(counts) * counts[c]) for c in counts.keys()}
                config.class_weights = class_weights
                
                # 创建样本权重
                sample_weights = [1.0 / counts[int(l)] for l in labels]
                train_sampler = WeightedRandomSampler(
                    sample_weights, num_samples=len(sample_weights), replacement=True
                )
                print(f"✅ Created WeightedRandomSampler: {dict(counts)}")
                print(f"   Class weights: {class_weights}")
        except Exception as e:
            print(f"⚠️ Failed to create weighted sampler: {e}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 打印数据集统计信息
    print(f"✅ ResNet Stage2 data loaders created:")
    print(f"   Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset):,} samples, {len(val_loader)} batches") 
    print(f"   Test:  {len(test_dataset):,} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试ResNet数据集
    print("Testing ResNet Stage2 Dataset...")
    
    # 由于需要实际数据集，这里只测试配置
    from configs.resnet_stage2_config import get_resnet18_config
    
    config = get_resnet18_config(
        data_path="../dataset",  # 假设路径
        batch_size=4
    )
    
    print(f"Config validation passed")
    print(f"Model info: {config.get_model_info()}")
    
    # 如果有实际数据路径，可以测试数据加载
    test_data_path = "../dataset"
    if os.path.exists(test_data_path):
        print(f"\nTesting with real data at {test_data_path}...")
        try:
            train_loader, val_loader, test_loader = create_resnet_stage2_data_loaders(config)
            print("✅ Data loaders created successfully!")
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
    else:
        print(f"⚠️ Test data path {test_data_path} not found, skipping data loading test")
    
    print("\n✅ ResNet Stage2 dataset test completed!")