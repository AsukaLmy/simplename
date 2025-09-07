#!/usr/bin/env python3
"""
Relation Network模式测试脚本
测试Relation Network数据集、模型和训练流程
"""

import os
import sys

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.stage2_config import Stage2Config
from utils.model_factory import create_stage2_model, create_stage2_loss
from utils.data_factory import create_stage2_data_loaders
from datasets.stage2_dataset import RelationStage2Dataset


def test_relation_config():
    """测试Relation Network配置"""
    print("🔧 Testing Relation Network Configuration...")
    
    config = Stage2Config(
        temporal_mode='relation',           # Relation模式
        use_geometric=True,
        use_hog=True,
        use_scene_context=True,
        fusion_strategy='concat',           # Simple Concatenation
        relation_hidden_dims=[64, 64],
        spatial_feature_dim=0,              # 暂时不使用额外空间特征
        dropout=0.2,
        batch_size=4,                       # 小批次便于测试
        data_path="../dataset"
    )
    
    config.validate()
    print(f"✅ Relation Config validated:")
    print(f"  Mode: {config.temporal_mode}")
    print(f"  Fusion strategy: {config.fusion_strategy}")
    print(f"  Input dim (for reference): {config.get_input_dim()}")
    print(f"  Relation hidden dims: {config.relation_hidden_dims}")
    
    return config


def test_relation_model(config):
    """测试Relation Network模型"""
    print(f"\n🧠 Testing Relation Network Model...")
    
    try:
        model = create_stage2_model(config)
        model_info = model.get_model_info()
        print(f"✅ Relation Network Model created:")
        print(f"  Type: {model_info['model_type']}")
        print(f"  Parameters: {model_info['trainable_params']:,}")
        print(f"  Person feature dim: {model_info['person_feature_dim']}")
        print(f"  Spatial feature dim: {model_info['spatial_feature_dim']}")
        print(f"  Fusion strategy: {model_info['fusion_strategy']}")
        
        # 测试前向传播
        batch_size = 2
        person_feature_dim = model_info['person_feature_dim']
        spatial_feature_dim = model_info['spatial_feature_dim']
        
        # 创建测试输入 (模拟RelationStage2Dataset的输出格式)
        test_person_A = torch.randn(batch_size, person_feature_dim)
        test_person_B = torch.randn(batch_size, person_feature_dim)
        test_spatial = torch.randn(batch_size, spatial_feature_dim) if spatial_feature_dim > 0 else torch.empty(batch_size, 0)
        
        print(f"\n📊 Testing forward pass:")
        print(f"  Person A input: {test_person_A.shape}")
        print(f"  Person B input: {test_person_B.shape}")
        print(f"  Spatial input: {test_spatial.shape}")
        
        with torch.no_grad():
            output = model(test_person_A, test_person_B, test_spatial)
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return model
        
    except Exception as e:
        print(f"❌ Relation Network model test failed: {e}")
        raise


def test_relation_dataset(config):
    """测试Relation Network数据集"""
    print(f"\n📚 Testing Relation Network Dataset...")
    
    try:
        # 创建小数据集进行测试
        dataset = RelationStage2Dataset(
            data_path=config.data_path,
            split='train',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 测试时不使用过采样
        )
        
        print(f"✅ Relation Dataset created:")
        print(f"  Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试样本加载
            sample = dataset[0]
            print(f"\n📋 Sample structure:")
            print(f"  Keys: {sample.keys()}")
            print(f"  Person A features shape: {sample['person_A_features'].shape}")
            print(f"  Person B features shape: {sample['person_B_features'].shape}")
            print(f"  Spatial features shape: {sample['spatial_features'].shape}")
            print(f"  Label: {sample['stage2_label'].item()}")
            print(f"  Person A/B IDs: {sample['person_A_id']}/{sample['person_B_id']}")
            print(f"  Frame ID: {sample['frame_id']}")
            
            # 测试多个样本
            print(f"\n📊 Testing multiple samples:")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"  Sample {i}: person_A={sample['person_A_features'].shape}, "
                      f"person_B={sample['person_B_features'].shape}, "
                      f"spatial={sample['spatial_features'].shape}, "
                      f"label={sample['stage2_label'].item()}")
            
            # 类别分布
            distribution = dataset.get_class_distribution()
            print(f"\n📈 Class distribution: {distribution}")
        
        return dataset
        
    except FileNotFoundError as e:
        print(f"⚠️  Dataset path not found: {e}")
        print("This is expected if the dataset doesn't exist")
        return None
    except Exception as e:
        print(f"❌ Relation Network dataset test failed: {e}")
        raise


def test_relation_data_loaders(config):
    """测试Relation Network数据加载器"""
    print(f"\n🔄 Testing Relation Network Data Loaders...")
    
    try:
        train_loader, val_loader, test_loader = create_stage2_data_loaders(config)
        
        print(f"✅ Data loaders created:")
        print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"  Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
        print(f"  Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
        
        # 测试批次加载
        if len(train_loader) > 0:
            print(f"\n🎯 Testing batch loading:")
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 2:  # 只测试前2个批次
                    break
                
                person_A_features = batch['person_A_features']  # [batch_size, person_feature_dim]
                person_B_features = batch['person_B_features']  # [batch_size, person_feature_dim]
                spatial_features = batch['spatial_features']    # [batch_size, spatial_feature_dim]
                labels = batch['stage2_label']                   # [batch_size]
                
                print(f"  Batch {batch_idx}:")
                print(f"    Person A features: {person_A_features.shape}")
                print(f"    Person B features: {person_B_features.shape}")
                print(f"    Spatial features: {spatial_features.shape}")
                print(f"    Labels: {labels.shape}, unique: {torch.unique(labels).tolist()}")
        
        return train_loader, val_loader, test_loader
        
    except FileNotFoundError as e:
        print(f"⚠️  Dataset path not found: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        raise


def test_relation_loss(config, model):
    """测试Relation Network损失函数"""
    print(f"\n💥 Testing Relation Network Loss Function...")
    
    try:
        criterion = create_stage2_loss(config)
        
        # 创建测试数据
        batch_size = 2
        model_info = model.get_model_info()
        person_feature_dim = model_info['person_feature_dim']
        spatial_feature_dim = model_info['spatial_feature_dim']
        
        test_person_A = torch.randn(batch_size, person_feature_dim)
        test_person_B = torch.randn(batch_size, person_feature_dim)
        test_spatial = torch.randn(batch_size, spatial_feature_dim) if spatial_feature_dim > 0 else torch.empty(batch_size, 0)
        test_labels = torch.randint(0, 3, (batch_size,))
        
        # 前向传播
        with torch.no_grad():
            logits = model(test_person_A, test_person_B, test_spatial)
            loss, loss_dict = criterion(logits, test_labels)
        
        print(f"✅ Loss computation successful:")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Loss details: {loss_dict}")
        
        return criterion
        
    except Exception as e:
        print(f"❌ Relation Network loss test failed: {e}")
        raise


def test_training_step(config, model, criterion, data_loader):
    """测试训练步骤"""
    print(f"\n🏃 Testing Training Step...")
    
    if data_loader is None or len(data_loader) == 0:
        print("⚠️  No data loader available for training test")
        return
    
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # 运行一个训练步骤
        batch = next(iter(data_loader))
        person_A_features = batch['person_A_features']
        person_B_features = batch['person_B_features']
        spatial_features = batch['spatial_features']
        labels = batch['stage2_label']
        
        print(f"  Training batch:")
        print(f"    Person A: {person_A_features.shape}")
        print(f"    Person B: {person_B_features.shape}")
        print(f"    Spatial: {spatial_features.shape}")
        print(f"    Labels: {labels.shape}")
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(person_A_features, person_B_features, spatial_features)
        loss, loss_dict = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step completed:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients: {'OK' if any(p.grad is not None for p in model.parameters()) else 'Missing'}")
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        raise


def main():
    """主测试函数"""
    print("Starting Relation Network Mode Comprehensive Test...")
    print("=" * 60)
    
    try:
        # 1. 测试配置
        config = test_relation_config()
        
        # 2. 测试模型
        model = test_relation_model(config)
        
        # 3. 测试损失函数
        criterion = test_relation_loss(config, model)
        
        # 4. 测试数据集
        dataset = test_relation_dataset(config)
        
        # 5. 测试数据加载器
        train_loader, val_loader, test_loader = test_relation_data_loaders(config)
        
        # 6. 测试训练步骤
        test_training_step(config, model, criterion, train_loader)
        
        print(f"\n" + "==" * 30)
        print("✅ Relation Network Mode Test Completed Successfully!")
        print("🚀 Ready for Relation Network training!")
        
        # 参数量对比
        basic_config = Stage2Config(temporal_mode='none')
        lstm_config = Stage2Config(temporal_mode='lstm')
        
        basic_params = 19779  # 从之前计算得出
        # lstm_params = XXX  # 需要实际测试得出
        relation_params = model.get_model_info()['trainable_params']
        
        print(f"\n📊 Parameter Comparison:")
        print(f"  Basic mode:    {basic_params:,} parameters")
        print(f"  Relation mode: {relation_params:,} parameters")
        print(f"  Ratio (vs Basic): {relation_params/basic_params:.1f}x")
        
        # 特征维度对比
        print(f"\n🔍 Feature Dimensions:")
        print(f"  Basic mode input: {basic_config.get_input_dim()}D (person_A + person_B + spatial combined)")
        print(f"  Relation mode:")
        model_info = model.get_model_info()
        print(f"    Person features: {model_info['person_feature_dim']}D (each person)")
        print(f"    Spatial features: {model_info['spatial_feature_dim']}D")
        print(f"    Fusion strategy: {model_info['fusion_strategy']}")
        
        concat_total = model_info['person_feature_dim'] * 2 + model_info['spatial_feature_dim']
        print(f"    Combined input: {concat_total}D (after concatenation)")
        
    except Exception as e:
        print(f"\n❌ Relation Network Mode Test Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()