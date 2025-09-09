#!/usr/bin/env python3
"""
调试验证准确率异常低的问题
"""
import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.stage2_config import Stage2Config
from utils.model_factory import create_stage2_model, create_stage2_loss
from utils.data_factory import create_stage2_data_loaders


def debug_model_state():
    """调试模型在train/eval状态下的行为差异"""
    print("Debugging model state behavior...")
    
    # 创建配置
    config = Stage2Config(
        temporal_mode='relation',
        data_path="../dataset",
        batch_size=4,  # 小批次便于调试
        frame_interval=100
    )
    
    # 创建模型和数据
    device = torch.device('cpu')  # 使用CPU便于调试
    model = create_stage2_model(config).to(device)
    criterion = create_stage2_loss(config).to(device)
    
    try:
        train_loader, val_loader, test_loader = create_stage2_data_loaders(config)
        
        if len(train_loader) == 0:
            print("ERROR: No training data available")
            return
            
        # 获取一个批次
        batch = next(iter(train_loader))
        person_A_features = batch['person_A_features'].to(device)
        person_B_features = batch['person_B_features'].to(device)
        spatial_features = batch['spatial_features'].to(device)
        targets = batch['stage2_label'].to(device)
        
        print(f"Batch info:")
        print(f"  Person A: {person_A_features.shape}")
        print(f"  Person B: {person_B_features.shape}")
        print(f"  Spatial: {spatial_features.shape}")
        print(f"  Targets: {targets}")
        print(f"  Unique labels: {torch.unique(targets)}")
        
        # 测试训练模式
        model.train()
        with torch.no_grad():
            outputs_train = model(person_A_features, person_B_features, spatial_features)
            probs_train = torch.softmax(outputs_train, dim=1)
            pred_train = torch.argmax(outputs_train, dim=1)
        
        # 测试评估模式
        model.eval()
        with torch.no_grad():
            outputs_eval = model(person_A_features, person_B_features, spatial_features)
            probs_eval = torch.softmax(outputs_eval, dim=1)
            pred_eval = torch.argmax(outputs_eval, dim=1)
        
        print(f"\n Model behavior comparison:")
        print(f"Training mode predictions: {pred_train.tolist()}")
        print(f"Eval mode predictions:     {pred_eval.tolist()}")
        print(f"Targets:                   {targets.tolist()}")
        
        print(f"\nTraining mode probs:")
        for i in range(len(probs_train)):
            print(f"  Sample {i}: {probs_train[i].tolist()}")
        
        print(f"\nEval mode probs:")
        for i in range(len(probs_eval)):
            print(f"  Sample {i}: {probs_eval[i].tolist()}")
        
        # 检查概率分布差异
        prob_diff = torch.abs(probs_train - probs_eval).max().item()
        print(f"\nMax probability difference: {prob_diff:.6f}")
        
        if prob_diff > 0.01:
            print("WARNING:  WARNING: Significant difference between train/eval modes!")
            print("   This suggests Dropout is affecting validation results")
        else:
            print("OK: Train/eval modes produce similar outputs")
            
    except Exception as e:
        print(f"ERROR: Error during debugging: {e}")
        import traceback
        traceback.print_exc()


def debug_feature_quality():
    """调试特征质量"""
    print("\n Debugging feature quality...")
    
    config = Stage2Config(
        temporal_mode='relation',
        data_path="../dataset",
        batch_size=8,
        frame_interval=100
    )
    
    try:
        train_loader, val_loader, test_loader = create_stage2_data_loaders(config)
        
        # 收集一些样本的特征
        features_by_class = {0: [], 1: [], 2: []}
        sample_count = 0
        
        for batch in train_loader:
            person_A_features = batch['person_A_features']
            person_B_features = batch['person_B_features'] 
            spatial_features = batch['spatial_features']
            targets = batch['stage2_label']
            
            # 拼接特征 (模拟Relation Network的concat操作)
            combined_features = torch.cat([person_A_features, person_B_features, spatial_features], dim=1)
            
            for i in range(len(targets)):
                label = targets[i].item()
                features_by_class[label].append(combined_features[i])
                sample_count += 1
                
            if sample_count > 50:  # 只检查前50个样本
                break
        
        print(f"Collected {sample_count} samples")
        
        # 计算各类别的特征统计
        for class_id in [0, 1, 2]:
            if len(features_by_class[class_id]) > 0:
                class_features = torch.stack(features_by_class[class_id])
                mean_feat = class_features.mean(dim=0)
                std_feat = class_features.std(dim=0)
                
                print(f"\nClass {class_id} ({len(features_by_class[class_id])} samples):")
                print(f"  Feature mean: {mean_feat[:5].tolist()}... (first 5)")
                print(f"  Feature std:  {std_feat[:5].tolist()}... (first 5)")
                print(f"  Mean magnitude: {mean_feat.norm().item():.4f}")
                print(f"  Std magnitude:  {std_feat.mean().item():.4f}")
                
        # 检查特征是否有明显差异
        if len(features_by_class[0]) > 0 and len(features_by_class[1]) > 0:
            feat0 = torch.stack(features_by_class[0]).mean(dim=0)
            feat1 = torch.stack(features_by_class[1]).mean(dim=0)
            distance = torch.dist(feat0, feat1).item()
            print(f"\nDistance between class 0 and 1 features: {distance:.4f}")
            
            if distance < 0.1:
                print("WARNING:  WARNING: Very small distance between class features!")
                print("   Features might not be discriminative enough")
            else:
                print("OK: Features show reasonable class separation")
                
    except Exception as e:
        print(f"ERROR: Error during feature debugging: {e}")
        import traceback
        traceback.print_exc()


def debug_loss_and_gradients():
    """调试损失和梯度"""
    print("\n Debugging loss and gradients...")
    
    config = Stage2Config(
        temporal_mode='relation',
        data_path="../dataset",
        batch_size=4,
        frame_interval=100
    )
    
    device = torch.device('cpu')
    model = create_stage2_model(config).to(device)
    criterion = create_stage2_loss(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        train_loader, _, _ = create_stage2_data_loaders(config)
        
        if len(train_loader) == 0:
            print("ERROR: No training data available")
            return
            
        batch = next(iter(train_loader))
        person_A_features = batch['person_A_features'].to(device)
        person_B_features = batch['person_B_features'].to(device)
        spatial_features = batch['spatial_features'].to(device)
        targets = batch['stage2_label'].to(device)
        
        print(f"Target distribution: {torch.bincount(targets)}")
        
        # 前向传播
        model.train()
        outputs = model(person_A_features, person_B_features, spatial_features)
        loss, loss_dict = criterion(outputs, targets)
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Loss details: {loss_dict}")
        
        # 检查输出分布
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        
        print(f"Predictions: {predictions.tolist()}")
        print(f"Targets:     {targets.tolist()}")
        print(f"Accuracy: {(predictions == targets).float().mean().item():.4f}")
        
        print(f"\nOutput logits:")
        for i in range(len(outputs)):
            print(f"  Sample {i}: {outputs[i].tolist()}")
            
        # 反向传播检查梯度
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                if param_count <= 3:  # 只显示前3个参数
                    print(f"Gradient norm for {name}: {grad_norm:.6f}")
        
        avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
        print(f"Average gradient norm: {avg_grad_norm:.6f}")
        
        if avg_grad_norm < 1e-6:
            print("WARNING:  WARNING: Very small gradients! Model might not be learning")
        else:
            print("OK: Gradients look reasonable")
            
    except Exception as e:
        print(f"ERROR: Error during loss/gradient debugging: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主调试函数"""
    print("Starting validation accuracy bug debugging...")
    print("=" * 60)
    
    # 1. 检查模型状态行为
    debug_model_state()
    
    # 2. 检查特征质量
    debug_feature_quality()
    
    # 3. 检查损失和梯度
    debug_loss_and_gradients()
    
    print("\n" + "=" * 60)
    print("Debugging completed. Check the output above for issues.")


if __name__ == '__main__':
    main()