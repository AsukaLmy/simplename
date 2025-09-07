#!/usr/bin/env python3
"""
LSTM模式测试脚本
测试LSTM数据集、模型和训练流程
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
from datasets.stage2_dataset import LSTMStage2Dataset


def test_lstm_config():
    """测试LSTM配置"""
    print("🔧 Testing LSTM Configuration...")
    
    config = Stage2Config(
        temporal_mode='lstm',           # LSTM模式
        use_geometric=True,
        use_hog=True,
        use_scene_context=True,
        sequence_length=5,              # 5帧序列
        lstm_hidden_dim=64,
        lstm_layers=2,
        bidirectional=True,
        hidden_dims=[64, 32],
        dropout=0.2,
        batch_size=4,                   # 小批次便于测试
        data_path="../dataset"
    )
    
    config.validate()
    print(f"✅ LSTM Config validated:")
    print(f"  Mode: {config.temporal_mode}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Input dim: {config.get_input_dim()}")
    print(f"  LSTM config: {config.lstm_layers} layers, {config.lstm_hidden_dim} hidden, bidirectional={config.bidirectional}")
    
    return config


def test_lstm_model(config):
    """测试LSTM模型"""
    print(f"\n🧠 Testing LSTM Model...")
    
    try:
        model = create_stage2_model(config)
        model_info = model.get_model_info()
        print(f"✅ LSTM Model created:")
        print(f"  Type: {model_info['model_type']}")
        print(f"  Parameters: {model_info['trainable_params']:,}")
        print(f"  Feature dim: {model_info['feature_dim']}")
        print(f"  Sequence length: {model_info['sequence_length']}")
        
        # 测试前向传播
        batch_size = 2
        seq_len = config.sequence_length
        feat_dim = config.get_input_dim()
        
        # 创建测试输入 [batch_size, sequence_length, feature_dim]
        test_input = torch.randn(batch_size, seq_len, feat_dim)
        print(f"\n📊 Testing forward pass:")
        print(f"  Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return model
        
    except Exception as e:
        print(f"❌ LSTM model test failed: {e}")
        raise


def test_lstm_dataset(config):
    """测试LSTM数据集"""
    print(f"\n📚 Testing LSTM Dataset...")
    
    try:
        # 创建小数据集进行测试
        dataset = LSTMStage2Dataset(
            data_path=config.data_path,
            split='train',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            sequence_length=config.sequence_length,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 测试时不使用过采样
        )
        
        print(f"✅ LSTM Dataset created:")
        print(f"  Total sequences: {len(dataset)}")
        
        if len(dataset) > 0:
            # 测试样本加载
            sample = dataset[0]
            print(f"\n📋 Sample structure:")
            print(f"  Keys: {sample.keys()}")
            print(f"  Sequences shape: {sample['sequences'].shape}")
            print(f"  Label: {sample['stage2_label'].item()}")
            print(f"  Group key: {sample['group_key']}")
            print(f"  Frame range: {sample['start_frame']} - {sample['end_frame']}")
            
            # 测试多个样本
            print(f"\n📊 Testing multiple samples:")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"  Sample {i}: sequences={sample['sequences'].shape}, label={sample['stage2_label'].item()}")
            
            # 类别分布
            distribution = dataset.get_class_distribution()
            print(f"\n📈 Class distribution: {distribution}")
        
        return dataset
        
    except FileNotFoundError as e:
        print(f"⚠️  Dataset path not found: {e}")
        print("This is expected if the dataset doesn't exist")
        return None
    except Exception as e:
        print(f"❌ LSTM dataset test failed: {e}")
        raise


def test_lstm_data_loaders(config):
    """测试LSTM数据加载器"""
    print(f"\n🔄 Testing LSTM Data Loaders...")
    
    try:
        train_loader, val_loader, test_loader = create_stage2_data_loaders(config)
        
        print(f"✅ Data loaders created:")
        print(f"  Train: {len(train_loader.dataset)} sequences, {len(train_loader)} batches")
        print(f"  Val: {len(val_loader.dataset)} sequences, {len(val_loader)} batches")
        print(f"  Test: {len(test_loader.dataset)} sequences, {len(test_loader)} batches")
        
        # 测试批次加载
        if len(train_loader) > 0:
            print(f"\n🎯 Testing batch loading:")
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 2:  # 只测试前2个批次
                    break
                
                sequences = batch['sequences']  # [batch_size, seq_len, feat_dim]
                labels = batch['stage2_label']   # [batch_size]
                
                print(f"  Batch {batch_idx}:")
                print(f"    Sequences: {sequences.shape}")
                print(f"    Labels: {labels.shape}, unique: {torch.unique(labels).tolist()}")
        
        return train_loader, val_loader, test_loader
        
    except FileNotFoundError as e:
        print(f"⚠️  Dataset path not found: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        raise


def test_lstm_loss(config, model):
    """测试LSTM损失函数"""
    print(f"\n💥 Testing LSTM Loss Function...")
    
    try:
        criterion = create_stage2_loss(config)
        
        # 创建测试数据
        batch_size = 2
        seq_len = config.sequence_length
        feat_dim = config.get_input_dim()
        
        test_sequences = torch.randn(batch_size, seq_len, feat_dim)
        test_labels = torch.randint(0, 3, (batch_size,))
        
        # 前向传播
        with torch.no_grad():
            logits = model(test_sequences)
            loss, loss_dict = criterion(logits, test_labels)
        
        print(f"✅ Loss computation successful:")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Loss details: {loss_dict}")
        
        return criterion
        
    except Exception as e:
        print(f"❌ LSTM loss test failed: {e}")
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
        sequences = batch['sequences']
        labels = batch['stage2_label']
        
        print(f"  Training batch:")
        print(f"    Input: {sequences.shape}")
        print(f"    Labels: {labels.shape}")
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(sequences)
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
    print("🔍 Starting LSTM Mode Comprehensive Test...")
    print("=" * 60)
    
    try:
        # 1. 测试配置
        config = test_lstm_config()
        
        # 2. 测试模型
        model = test_lstm_model(config)
        
        # 3. 测试损失函数
        criterion = test_lstm_loss(config, model)
        
        # 4. 测试数据集
        dataset = test_lstm_dataset(config)
        
        # 5. 测试数据加载器
        train_loader, val_loader, test_loader = test_lstm_data_loaders(config)
        
        # 6. 测试训练步骤
        test_training_step(config, model, criterion, train_loader)
        
        print(f"\n" + "=" * 60)
        print("✅ LSTM Mode Test Completed Successfully!")
        print("🚀 Ready for LSTM training!")
        
        # 参数量对比
        basic_config = Stage2Config(temporal_mode='none')
        basic_params = 19779  # 从之前计算得出
        lstm_params = model.get_model_info()['trainable_params']
        
        print(f"\n📊 Parameter Comparison:")
        print(f"  Basic mode: {basic_params:,} parameters")
        print(f"  LSTM mode:  {lstm_params:,} parameters")
        print(f"  Ratio: {lstm_params/basic_params:.1f}x")
        
    except Exception as e:
        print(f"\n❌ LSTM Mode Test Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()