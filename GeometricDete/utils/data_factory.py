#!/usr/bin/env python3
"""
Data Factory for Stage2 Behavior Classification
Creates appropriate datasets and data loaders based on configuration
"""

from torch.utils.data import DataLoader
from typing import Tuple
import os

from configs.stage2_config import Stage2Config


def create_stage2_data_loaders(config: Stage2Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建Stage2数据加载器
    
    Args:
        config: Stage2配置对象
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    
    if config.temporal_mode == 'none':
        # Basic模式 - 使用重构的数据集
        from datasets.stage2_dataset import BasicStage2Dataset
        
        print(f"🔄 Creating Basic mode datasets...")
        print(f"   Features: {'Geometric(7)' if config.use_geometric else ''}"
              f"{' + HoG(64)' if config.use_hog else ''}"
              f"{' + Scene(1)' if config.use_scene_context else ''}")
        print(f"   Frame interval: {config.frame_interval}")
        
        # 创建数据集
        train_dataset = BasicStage2Dataset(
            data_path=config.data_path,
            split='train',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            frame_interval=config.frame_interval,
            use_oversampling=True  # 训练集使用过采样
        )
        
        val_dataset = BasicStage2Dataset(
            data_path=config.data_path,
            split='val',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 验证集不使用过采样
        )
        
        test_dataset = BasicStage2Dataset(
            data_path=config.data_path,
            split='test',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 测试集不使用过采样
        )
        
    elif config.temporal_mode == 'lstm':
        # LSTM模式 - 使用时序数据集
        from datasets.stage2_dataset import LSTMStage2Dataset
        
        print(f"🔄 Creating LSTM mode datasets...")
        print(f"   Features: {'Geometric(7)' if config.use_geometric else ''}"
              f"{' + HoG(64)' if config.use_hog else ''}"
              f"{' + Scene(1)' if config.use_scene_context else ''}")
        print(f"   Sequence length: {config.sequence_length}")
        print(f"   Frame interval: {config.frame_interval}")
        
        # 创建时序数据集
        train_dataset = LSTMStage2Dataset(
            data_path=config.data_path,
            split='train',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            sequence_length=config.sequence_length,
            frame_interval=config.frame_interval,
            use_oversampling=True  # 训练集使用时序过采样
        )
        
        val_dataset = LSTMStage2Dataset(
            data_path=config.data_path,
            split='val',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            sequence_length=config.sequence_length,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 验证集不使用过采样
        )
        
        test_dataset = LSTMStage2Dataset(
            data_path=config.data_path,
            split='test',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            sequence_length=config.sequence_length,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 测试集不使用过采样
        )
        
    elif config.temporal_mode == 'relation':
        # Relation Network模式 - 使用关系网络数据集
        from datasets.stage2_dataset import RelationStage2Dataset
        
        print(f"🔄 Creating Relation Network mode datasets...")
        print(f"   Features: Person={'HoG(32)' if config.use_hog else 'None'}")
        print(f"   Spatial: {'Geometric(7)' if config.use_geometric else ''}"
              f"{' + Scene(1)' if config.use_scene_context else ''}")
        print(f"   Fusion strategy: {config.fusion_strategy}")
        print(f"   Frame interval: {config.frame_interval}")
        
        # 创建关系网络数据集
        train_dataset = RelationStage2Dataset(
            data_path=config.data_path,
            split='train',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            frame_interval=config.frame_interval,
            use_oversampling=True  # 训练集使用过采样
        )
        
        val_dataset = RelationStage2Dataset(
            data_path=config.data_path,
            split='val',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 验证集不使用过采样
        )
        
        test_dataset = RelationStage2Dataset(
            data_path=config.data_path,
            split='test',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 测试集不使用过采样
        )
        
    else:
        raise ValueError(f"Unknown temporal_mode: {config.temporal_mode}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True  # 确保批次大小一致
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
    print(f"✅ Data loaders created:")
    print(f"   Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset):,} samples, {len(test_loader)} batches")
    print(f"   Feature dimension: {config.get_input_dim()}D")
    
    return train_loader, val_loader, test_loader


def get_dataset_statistics(data_loaders: Tuple[DataLoader, DataLoader, DataLoader]) -> dict:
    """
    获取数据集统计信息
    
    Args:
        data_loaders: (train_loader, val_loader, test_loader)
        
    Returns:
        dict: 统计信息
    """
    train_loader, val_loader, test_loader = data_loaders
    
    # 获取类别分布
    def get_class_distribution(dataset):
        if hasattr(dataset, 'get_class_distribution'):
            return dataset.get_class_distribution()
        else:
            return {"message": "Class distribution not available"}
    
    train_dist = get_class_distribution(train_loader.dataset)
    val_dist = get_class_distribution(val_loader.dataset)
    test_dist = get_class_distribution(test_loader.dataset)
    
    # 获取样本统计
    def get_sample_info(dataset):
        try:
            sample = dataset[0]
            
            # 处理不同类型的数据集输出格式
            if 'features' in sample:
                # Basic/LSTM模式
                feature_shape = sample['features'].shape
                return {
                    'sample_count': len(dataset),
                    'feature_shape': list(feature_shape),
                    'feature_dim': feature_shape[0] if len(feature_shape) == 1 else feature_shape[-1]
                }
            elif 'sequences' in sample:
                # LSTM模式
                seq_shape = sample['sequences'].shape
                return {
                    'sample_count': len(dataset),
                    'sequence_shape': list(seq_shape),
                    'feature_dim': seq_shape[-1] if len(seq_shape) > 1 else seq_shape[0]
                }
            elif 'person_A_features' in sample and 'person_B_features' in sample:
                # Relation Network模式
                person_A_shape = sample['person_A_features'].shape
                person_B_shape = sample['person_B_features'].shape
                spatial_shape = sample['spatial_features'].shape if sample['spatial_features'].numel() > 0 else [0]
                return {
                    'sample_count': len(dataset),
                    'person_A_shape': list(person_A_shape),
                    'person_B_shape': list(person_B_shape), 
                    'spatial_shape': list(spatial_shape),
                    'person_feature_dim': person_A_shape[0] if len(person_A_shape) == 1 else person_A_shape[-1],
                    'spatial_feature_dim': spatial_shape[0] if len(spatial_shape) == 1 else spatial_shape[-1]
                }
            else:
                return {'error': 'Unknown dataset output format'}
        except Exception as e:
            return {'error': str(e)}
    
    train_info = get_sample_info(train_loader.dataset)
    val_info = get_sample_info(val_loader.dataset)
    test_info = get_sample_info(test_loader.dataset)
    
    statistics = {
        'train': {
            'distribution': train_dist,
            'info': train_info
        },
        'val': {
            'distribution': val_dist,
            'info': val_info
        },
        'test': {
            'distribution': test_dist,
            'info': test_info
        },
        'total_samples': train_info.get('sample_count', 0) + val_info.get('sample_count', 0) + test_info.get('sample_count', 0)
    }
    
    return statistics


def print_dataset_summary(config: Stage2Config, data_loaders: Tuple[DataLoader, DataLoader, DataLoader]):
    """
    打印数据集摘要信息
    
    Args:
        config: 配置对象
        data_loaders: 数据加载器元组
    """
    print("\n" + "="*60)
    print("STAGE2 DATASET SUMMARY")
    print("="*60)
    
    print(f"Mode: {config.temporal_mode.upper()}")
    print(f"Data path: {config.data_path}")
    print(f"Frame interval: {config.frame_interval}")
    print(f"Batch size: {config.batch_size}")
    
    # 特征配置
    features = []
    if config.use_geometric:
        features.append("Geometric(7)")
    if config.use_hog:
        features.append("HoG(64)")
    if config.use_scene_context:
        features.append("Scene(1)")
    
    print(f"Features: {' + '.join(features)}")
    print(f"Total feature dimension: {config.get_input_dim()}D")
    
    # 数据统计
    try:
        stats = get_dataset_statistics(data_loaders)
        train_loader, val_loader, test_loader = data_loaders
        
        print(f"\nDataset Sizes:")
        print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
        print(f"  Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")  
        print(f"  Test:  {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")
        print(f"  Total: {stats['total_samples']:,} samples")
        
        # 类别分布 (如果可用)
        train_dist = stats['train']['distribution']
        if 'class_counts' in train_dist:
            print(f"\nTrain Class Distribution:")
            class_names = train_dist.get('class_names', [f'Class_{i}' for i in range(3)])
            for i, (class_id, count) in enumerate(train_dist['class_counts'].items()):
                class_name = class_names[i] if i < len(class_names) else f'Class_{class_id}'
                percentage = 100 * count / train_dist['total'] if train_dist['total'] > 0 else 0
                print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
                
    except Exception as e:
        print(f"\nWarning: Could not gather detailed statistics: {e}")
    
    print("="*60)


if __name__ == '__main__':
    # 测试数据工厂
    print("Testing Data Factory...")
    
    # 由于需要实际数据集，这里只测试配置验证
    from configs.stage2_config import Stage2Config
    
    print("\n1. Testing Basic mode configuration...")
    config_basic = Stage2Config(
        temporal_mode='none',
        data_path="../dataset",  # 假设路径
        use_geometric=True,
        use_hog=True,
        use_scene_context=True,
        batch_size=32,
        frame_interval=1
    )
    
    print(f"Config validation:")
    config_basic.validate()
    print(f"  Mode: {config_basic.temporal_mode}")
    print(f"  Input dimension: {config_basic.get_input_dim()}")
    print(f"  Features: Geometric={config_basic.use_geometric}, HoG={config_basic.use_hog}, Scene={config_basic.use_scene_context}")
    
    print(f"\n2. Testing frame interval configuration...")
    config_sparse = Stage2Config(
        temporal_mode='none',
        data_path="../dataset",
        frame_interval=10,  # 每10帧采样
        batch_size=64
    )
    
    config_sparse.validate()
    print(f"  Frame interval: {config_sparse.frame_interval}")
    print(f"  Expected sample reduction: ~90%")
    
    # 注意：实际的数据加载器创建需要存在的数据集
    print(f"\n3. Data loader creation would require existing dataset at: {config_basic.data_path}")
    print("   Use create_stage2_data_loaders() when dataset is available")
    
    print("\n✅ Data factory test completed!")