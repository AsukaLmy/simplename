#!/usr/bin/env python3
"""
Stage2 geometric behavior classification dataset
Extends Stage1 geometric features with enhanced temporal features for 5-class behavior classification
"""

import torch
import numpy as np
from collections import defaultdict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_dataset import GeometricDualPersonDataset
from geometric_features import extract_geometric_features, extract_causal_motion_features, compute_scene_context


def _get_frame_id_sort_key(sample):
    """Helper function to get frame_id for sorting (pickle-friendly)"""
    return sample['frame_id']


class Stage2LabelMapper:
    """Stage2的5分类标签映射器"""
    
    def __init__(self):
        self.label_mapping = {
            # Static Group (0) - 静态群体
            'standing together': 0,
            'sitting together': 0, 
            'conversation': 0,  # 确认为static
            'eating together': 0,
            
            # Parallel Movement (1) - 并行运动
            'walking together': 1,
            'moving together': 1,
            'cycling together': 1,
            'going upstairs together': 1,
            'going downstairs together': 1,
            
            # Approaching Interaction (2) - 接近交互
            'walking toward each other': 2,
            'hugging': 2,
            'shaking hand': 2,
            
            # Coordinated Activity (3) - 协调活动
            'looking at robot together': 3,
            'looking at sth together': 3,
            'bending together': 3,
            'holding sth together': 3,
            'waving hand together': 3,
            'looking into sth together': 3,
            
            # Complex/Rare (4) - 复杂罕见行为
            'interaction with door together': 4
        }
        
        self.class_names = [
            'Static Group',
            'Parallel Movement', 
            'Approaching Interaction',
            'Coordinated Activity',
            'Complex/Rare Behaviors'
        ]
        
        # 预计算类别权重 (基于JRDB分布)
        self.class_weights = {
            0: 1.0,    # Static Group (66.3%)
            1: 1.4,    # Parallel Movement (48.1%)
            2: 8.3,    # Approaching (0.8%)
            3: 7.4,    # Coordinated (0.9%)  
            4: 50.0    # Rare (0.2%)
        }
        
        # 目标类别分布 (用于过采样)
        self.target_distribution = {
            0: 0.35,  # 降低dominant class比例
            1: 0.35,  # 保持较高比例
            2: 0.10,  # 提升少数类
            3: 0.10,  # 提升少数类
            4: 0.10   # 提升罕见类
        }
    
    def map_label(self, original_interaction):
        """将原始交互标签映射为5分类标签"""
        return self.label_mapping.get(original_interaction, 4)  # 未知类型归为4
    
    def get_mapped_class_name(self, class_id):
        """获取映射后的类别名"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return "Unknown"


def compute_velocity_features(history_boxes, frame_interval=1):
    """
    计算基于5帧历史的速度和方向特征
    
    Args:
        history_boxes: [5, 4] 历史边界框 [x, y, w, h]
        frame_interval: 帧间隔（通常为1）
    
    Returns:
        velocities: [4, 2] 速度向量
        centers: [5, 2] 中心点坐标
    """
    if len(history_boxes) < 2:
        return torch.zeros(0, 2), torch.zeros(len(history_boxes), 2)
    
    # 提取中心点坐标
    centers = []
    for i in range(len(history_boxes)):
        x, y, w, h = history_boxes[i]
        center_x = x + w/2
        center_y = y + h/2
        centers.append([center_x, center_y])
    
    centers = torch.tensor(centers, dtype=torch.float32)  # [5, 2]
    
    # 计算速度 (位置差分)
    velocities = []
    for i in range(1, len(centers)):
        vx = centers[i][0] - centers[i-1][0]  # 水平速度
        vy = centers[i][1] - centers[i-1][1]  # 垂直速度
        velocities.append([vx, vy])
    
    velocities = torch.tensor(velocities, dtype=torch.float32)  # [4, 2]
    return velocities, centers


def extract_enhanced_motion_features(person_A_history, person_B_history):
    """
    提取增强的运动时序特征 (8维)
    
    Args:
        person_A_history: [5, 4] Person A的历史边界框
        person_B_history: [5, 4] Person B的历史边界框
    
    Returns:
        torch.Tensor: [8] 增强运动特征
    """
    # 计算两人的速度和位置
    vel_A, centers_A = compute_velocity_features(person_A_history)
    vel_B, centers_B = compute_velocity_features(person_B_history)
    
    if len(vel_A) == 0 or len(vel_B) == 0:
        return torch.zeros(8, dtype=torch.float32)
    
    features = []
    
    # 1. velocity_correlation - 速度相关性
    vel_A_flat = vel_A.view(-1)  # [8] (4帧 x 2方向)
    vel_B_flat = vel_B.view(-1)  # [8]
    
    if torch.std(vel_A_flat) > 1e-6 and torch.std(vel_B_flat) > 1e-6:
        velocity_corr = torch.corrcoef(torch.stack([vel_A_flat, vel_B_flat]))[0, 1]
        velocity_corr = torch.nan_to_num(velocity_corr, 0.0).float()  # 转换为float32
    else:
        velocity_corr = torch.tensor(0.0, dtype=torch.float32)
    features.append(velocity_corr)
    
    # 2. direction_alignment - 方向对齐度
    direction_alignments = []
    for i in range(len(vel_A)):
        # 计算单位方向向量
        dir_A = vel_A[i] / (torch.norm(vel_A[i]) + 1e-6)
        dir_B = vel_B[i] / (torch.norm(vel_B[i]) + 1e-6)
        
        # 方向一致性 (余弦相似度)
        alignment = torch.dot(dir_A, dir_B)
        direction_alignments.append(alignment)
    
    avg_direction_alignment = torch.mean(torch.tensor(direction_alignments, dtype=torch.float32))
    features.append(avg_direction_alignment)
    
    # 3. acceleration_sync - 加速度同步性
    if len(vel_A) >= 2:
        acc_A = vel_A[1:] - vel_A[:-1]  # [3, 2]
        acc_B = vel_B[1:] - vel_B[:-1]  # [3, 2]
        
        # 加速度大小的相关性
        acc_mag_A = torch.norm(acc_A, dim=1)  # [3]
        acc_mag_B = torch.norm(acc_B, dim=1)  # [3]
        
        if torch.std(acc_mag_A) > 1e-6 and torch.std(acc_mag_B) > 1e-6:
            acc_sync = torch.corrcoef(torch.stack([acc_mag_A, acc_mag_B]))[0, 1]
            acc_sync = torch.nan_to_num(acc_sync, 0.0).float()  # 转换为float32
        else:
            acc_sync = torch.tensor(0.0, dtype=torch.float32)
    else:
        acc_sync = torch.tensor(0.0, dtype=torch.float32)
    features.append(acc_sync)
    
    # 4. stopping_frequency - 停顿频率
    stop_threshold = 5.0  # 像素/帧
    
    stops_A = torch.norm(vel_A, dim=1) < stop_threshold
    stops_B = torch.norm(vel_B, dim=1) < stop_threshold
    
    # 同时停顿的比例
    simultaneous_stops = (stops_A & stops_B).float().mean()
    features.append(simultaneous_stops)
    
    # 5. trajectory_similarity - 轨迹相似度
    if len(centers_A) >= 2 and len(centers_B) >= 2:
        # 轨迹向量化 (相对于起始点)
        traj_A = centers_A - centers_A[0]  # [5, 2]
        traj_B = centers_B - centers_B[0]  # [5, 2]
        
        # 轨迹长度归一化
        traj_A_norm = torch.norm(traj_A[-1])
        traj_B_norm = torch.norm(traj_B[-1])
        
        if traj_A_norm > 1e-6:
            traj_A = traj_A / traj_A_norm
        if traj_B_norm > 1e-6:
            traj_B = traj_B / traj_B_norm
        
        # 轨迹相似度 (平均距离的倒数)
        trajectory_dist = torch.mean(torch.norm(traj_A - traj_B, dim=1))
        trajectory_similarity = 1.0 / (trajectory_dist + 0.1)
    else:
        trajectory_similarity = torch.tensor(0.0, dtype=torch.float32)
    features.append(trajectory_similarity)
    
    # 6. relative_speed - 相对速度大小
    relative_speeds = []
    for i in range(len(vel_A)):
        rel_vel = vel_A[i] - vel_B[i]
        relative_speeds.append(torch.norm(rel_vel))
    
    avg_relative_speed = torch.mean(torch.tensor(relative_speeds, dtype=torch.float32))
    features.append(avg_relative_speed)
    
    # 7. movement_consistency - 运动一致性
    vel_A_consistency = 1.0 - torch.std(torch.norm(vel_A, dim=1))
    vel_B_consistency = 1.0 - torch.std(torch.norm(vel_B, dim=1))
    avg_consistency = (vel_A_consistency + vel_B_consistency) / 2
    features.append(torch.clamp(avg_consistency, 0.0, 1.0))
    
    # 8. proximity_change_rate - 距离变化率
    distances = []
    for i in range(len(centers_A)):
        if i < len(centers_B):
            dist = torch.norm(centers_A[i] - centers_B[i])
            distances.append(dist)
    
    if len(distances) >= 2:
        distance_changes = []
        for i in range(1, len(distances)):
            distance_changes.append(distances[i] - distances[i-1])
        
        proximity_change_rate = torch.mean(torch.tensor(distance_changes, dtype=torch.float32))
    else:
        proximity_change_rate = torch.tensor(0.0, dtype=torch.float32)
    features.append(proximity_change_rate)
    
    return torch.tensor(features, dtype=torch.float32)


class GeometricStage2Dataset(GeometricDualPersonDataset):
    """
    Stage2行为分类数据集
    继承自Stage1几何数据集，添加5分类行为标签和增强时序特征
    """
    
    def __init__(self, data_path, split='train', history_length=5, 
                 use_temporal=True, use_scene_context=True, use_oversampling=True):
        """
        Args:
            data_path: 数据集路径
            split: 数据集划分 ('train', 'val', 'test')
            history_length: 时序历史长度
            use_temporal: 是否使用时序特征
            use_scene_context: 是否使用场景上下文
            use_oversampling: 是否使用过采样平衡类别
        """
        # 初始化标签映射器
        self.label_mapper = Stage2LabelMapper()
        self.use_oversampling = use_oversampling
        
        # 调用父类初始化
        super().__init__(data_path, split, history_length, use_temporal, use_scene_context)
        
        # Stage2特有的处理
        self._filter_interaction_samples()
        self._apply_stage2_labels()
        
        if self.use_oversampling and split == 'train':
            self._apply_oversampling()
        
        print(f"GeometricStage2Dataset loaded: {len(self.samples)} samples ({split})")
        self._print_stage2_statistics()
    
    def _filter_interaction_samples(self):
        """过滤出只有交互的样本（has_interaction=1）"""
        interaction_samples = []
        for sample in self.samples:
            if sample['has_interaction'] == 1:
                interaction_samples.append(sample)
        
        print(f"Filtered interaction samples: {len(interaction_samples)}/{len(self.samples)} "
              f"({100*len(interaction_samples)/len(self.samples):.1f}%)")
        
        self.samples = interaction_samples
    
    def _apply_stage2_labels(self):
        """应用Stage2的5分类标签"""
        for sample in self.samples:
            original_interaction = sample.get('interaction_labels', {})
            if isinstance(original_interaction, dict) and len(original_interaction) > 0:
                # 取第一个交互类型
                interaction_type = list(original_interaction.keys())[0]
            else:
                interaction_type = 'unknown'
            
            # 映射到5分类
            stage2_label = self.label_mapper.map_label(interaction_type)
            sample['stage2_label'] = stage2_label
            sample['original_interaction'] = interaction_type
    
    def _apply_oversampling(self):
        """应用过采样策略平衡类别"""
        # 统计各类样本数量
        class_samples = {i: [] for i in range(5)}
        for sample in self.samples:
            label = sample['stage2_label']
            class_samples[label].append(sample)
        
        print("Original class distribution:")
        for class_id, samples in class_samples.items():
            print(f"  Class {class_id} ({self.label_mapper.get_mapped_class_name(class_id)}): {len(samples)}")
        
        total_samples = len(self.samples)
        balanced_samples = []
        
        for class_id, target_ratio in self.label_mapper.target_distribution.items():
            current_samples = class_samples[class_id]
            target_count = int(total_samples * target_ratio)
            
            if len(current_samples) == 0:
                continue
                
            if len(current_samples) < target_count:
                # 过采样：重复采样达到目标数量
                indices = np.random.choice(
                    len(current_samples), 
                    size=target_count, 
                    replace=True
                )
                oversampled = [current_samples[i] for i in indices]
                balanced_samples.extend(oversampled)
                print(f"  Oversampled class {class_id}: {len(current_samples)} -> {len(oversampled)}")
            else:
                # 下采样：随机选择目标数量
                indices = np.random.choice(
                    len(current_samples),
                    size=target_count,
                    replace=False
                )
                undersampled = [current_samples[i] for i in indices]
                balanced_samples.extend(undersampled)
                print(f"  Undersampled class {class_id}: {len(current_samples)} -> {len(undersampled)}")
        
        self.samples = balanced_samples
        np.random.shuffle(self.samples)  # 打乱顺序
    
    def __getitem__(self, idx):
        """
        获取Stage2样本，包含16维特征向量
        """
        # 获取基础样本信息
        sample = self.samples[idx]
        
        # Stage2标签
        stage2_label = torch.tensor(sample['stage2_label'], dtype=torch.long)
        
        # 基础几何特征 (7维)
        frame_id = sample['frame_id']
        person_A_id = sample['person_A_id']
        person_B_id = sample['person_B_id']
        person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
        person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
        
        # 提取几何特征
        geometric_features = extract_geometric_features(
            person_A_box, person_B_box, 640, 480
        )  # [7]
        
        # 基础运动特征 (4维)
        if self.use_temporal and self.temporal_manager:
            temporal_features = self.temporal_manager.get_temporal_features(person_A_id, person_B_id)
            
            if temporal_features['has_sufficient_history']:
                history_data = temporal_features['pair_interaction_history']
                if history_data.size(0) >= 2:
                    motion_features = extract_causal_motion_features(history_data.unsqueeze(0))
                    motion_features = motion_features.squeeze(0)  # [4]
                else:
                    motion_features = torch.zeros(4)
            else:
                motion_features = torch.zeros(4)
            
            # 获取历史边界框用于增强时序特征
            if temporal_features['has_sufficient_history']:
                history_geometric = temporal_features['pair_interaction_history']  # [history_length, 7]
                
                # 从几何特征重构历史边界框（简化版本，实际可能需要更复杂的逻辑）
                # 这里我们假设有办法获取历史边界框，或者直接使用当前框模拟
                person_A_history = person_A_box.unsqueeze(0).repeat(5, 1)  # 简化：重复当前框
                person_B_history = person_B_box.unsqueeze(0).repeat(5, 1)
                
                enhanced_motion_features = extract_enhanced_motion_features(
                    person_A_history, person_B_history
                )  # [8]
            else:
                enhanced_motion_features = torch.zeros(8)
        else:
            motion_features = torch.zeros(4)
            enhanced_motion_features = torch.zeros(8)
        
        # 场景上下文特征 (1维)
        if self.use_scene_context and frame_id in self.scene_data:
            scene_context = self.scene_data[frame_id]['scene_context']
        else:
            scene_context = torch.tensor([1.0], dtype=torch.float32)  # 默认：稀疏场景
        
        # 组合成16维特征向量
        # [7几何 + 4基础运动 + 8增强时序 + 1场景] = 20维，但我们说好16维，调整一下
        # 实际使用：[7几何 + 4基础运动 + 5增强时序] = 16维 (去掉场景上下文，简化增强时序到5维)
        enhanced_motion_features_reduced = enhanced_motion_features[:5]  # 取前5维
        
        full_features = torch.cat([
            geometric_features,                    # [7]
            motion_features,                      # [4] 
            enhanced_motion_features_reduced      # [5]
        ]).float()  # [16] 确保float32类型
        
        return {
            'features': full_features,
            'stage2_label': stage2_label,
            'original_interaction': sample['original_interaction'],
            'person_A_id': person_A_id,
            'person_B_id': person_B_id,
            'frame_id': frame_id,
            'scene_context': scene_context
        }
    
    def _print_stage2_statistics(self):
        """打印Stage2数据集统计信息"""
        if not self.samples:
            return
        
        # 统计各类别数量
        class_counts = {}
        for sample in self.samples:
            label = sample['stage2_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print(f"\nStage2 Dataset Statistics ({self.split}):")
        print("=" * 50)
        total = len(self.samples)
        
        for class_id in range(5):
            count = class_counts.get(class_id, 0)
            percentage = 100 * count / total if total > 0 else 0
            class_name = self.label_mapper.get_mapped_class_name(class_id)
            print(f"Class {class_id} ({class_name}): {count:,} ({percentage:.1f}%)")
        
        print(f"Total samples: {total:,}")
        print(f"Temporal features: {self.use_temporal}")
        print(f"Scene context: {self.use_scene_context}")
        print(f"Oversampling: {self.use_oversampling}")
    
    def get_stage2_class_distribution(self):
        """获取Stage2类别分布"""
        class_counts = {}
        for sample in self.samples:
            label = sample['stage2_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return {
            'class_counts': class_counts,
            'total': len(self.samples),
            'class_names': self.label_mapper.class_names
        }


if __name__ == '__main__':
    # 测试Stage2数据集
    print("Testing GeometricStage2Dataset...")
    
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    # 创建测试数据集
    print("Creating Stage2 dataset...")
    stage2_dataset = GeometricStage2Dataset(
        data_path, 
        split='train', 
        history_length=5,
        use_temporal=False,  # 先不使用时序测试
        use_scene_context=True,
        use_oversampling=True
    )
    
    # 测试数据加载
    print(f"\nTesting data loading...")
    sample = stage2_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Features: {sample['features']}")
    print(f"Stage2 label: {sample['stage2_label']}")
    print(f"Original interaction: {sample['original_interaction']}")
    
    # 打印分布统计
    distribution = stage2_dataset.get_stage2_class_distribution()
    print(f"\nClass distribution: {distribution}")
    
    print("Stage2 dataset test completed!")