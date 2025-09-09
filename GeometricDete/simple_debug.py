#!/usr/bin/env python3
"""
简单的bug调试 - 检查模型在相同数据上的表现
"""
import os
import sys
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.stage2_config import Stage2Config
from utils.model_factory import create_stage2_model, create_stage2_loss
from utils.data_factory import create_stage2_data_loaders

def main():
    print("Simple debugging - checking model behavior")
    
    config = Stage2Config(
        temporal_mode='relation',
        data_path="../dataset",
        batch_size=8,
        frame_interval=100
    )
    
    device = torch.device('cpu')
    model = create_stage2_model(config).to(device)
    
    try:
        train_loader, val_loader, test_loader = create_stage2_data_loaders(config)
        
        print(f"Data loaders:")
        print(f"  Train: {len(train_loader)} batches, {len(train_loader.dataset)} samples")
        print(f"  Val: {len(val_loader)} batches, {len(val_loader.dataset)} samples")
        
        # 获取训练集的第一个批次
        train_batch = next(iter(train_loader))
        train_person_A = train_batch['person_A_features'][:4]  # 只取前4个样本
        train_person_B = train_batch['person_B_features'][:4]
        train_spatial = train_batch['spatial_features'][:4]
        train_targets = train_batch['stage2_label'][:4]
        
        print(f"Train batch targets: {train_targets.tolist()}")
        
        # 测试在训练模式下的输出
        model.train()
        with torch.no_grad():
            train_outputs = model(train_person_A, train_person_B, train_spatial)
            train_preds = torch.argmax(train_outputs, dim=1)
        
        # 测试在评估模式下的输出  
        model.eval()
        with torch.no_grad():
            eval_outputs = model(train_person_A, train_person_B, train_spatial)
            eval_preds = torch.argmax(eval_outputs, dim=1)
        
        print(f"Same data, train mode preds: {train_preds.tolist()}")
        print(f"Same data, eval mode preds:  {eval_preds.tolist()}")
        
        # 检查是否完全相同
        if torch.equal(train_preds, eval_preds):
            print("Train/eval predictions are identical - Dropout issue unlikely")
        else:
            print("Train/eval predictions differ - Dropout might be the issue!")
        
        # 现在训练几步看看能否学到什么
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = create_stage2_loss(config)
        
        print("\nTraining for 10 steps on the same batch...")
        for step in range(10):
            optimizer.zero_grad()
            outputs = model(train_person_A, train_person_B, train_spatial)
            loss, _ = criterion(outputs, train_targets)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == train_targets).float().mean()
                
            if step % 2 == 0:
                print(f"  Step {step}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
        
        # 最终测试
        model.eval()
        with torch.no_grad():
            final_outputs = model(train_person_A, train_person_B, train_spatial)
            final_preds = torch.argmax(final_outputs, dim=1)
            final_acc = (final_preds == train_targets).float().mean()
        
        print(f"Final predictions: {final_preds.tolist()}")
        print(f"Final accuracy on train batch: {final_acc.item():.4f}")
        
        if final_acc > 0.6:
            print("Model can learn the training batch - data/model OK")
        else:
            print("Model cannot even learn the training batch - fundamental problem!")
        
        # 测试验证集
        if len(val_loader) > 0:
            val_batch = next(iter(val_loader))
            val_person_A = val_batch['person_A_features'][:4]
            val_person_B = val_batch['person_B_features'][:4]  
            val_spatial = val_batch['spatial_features'][:4]
            val_targets = val_batch['stage2_label'][:4]
            
            print(f"\nVal batch targets: {val_targets.tolist()}")
            
            with torch.no_grad():
                val_outputs = model(val_person_A, val_person_B, val_spatial)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == val_targets).float().mean()
            
            print(f"Val predictions: {val_preds.tolist()}")
            print(f"Val accuracy: {val_acc.item():.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()