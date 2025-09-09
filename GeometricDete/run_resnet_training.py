#!/usr/bin/env python3
"""
ResNet Stage2 Training Runner
Quick start script for ResNet-based Stage2 behavior classification
"""

# 修复OpenMP错误
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import subprocess

def run_training(config_name="resnet18", **kwargs):
    """
    运行ResNet训练
    
    Args:
        config_name: 配置名称 (resnet18, resnet50)
        **kwargs: 额外的训练参数
    """
    
    # 基础命令
    cmd = [sys.executable, "train_resnet_stage2.py"]
    
    # 预定义配置
    configs = {
        "resnet18": {
            "backbone": "resnet18",
            "visual_dim": 256,
            "batch_size": 16,
            "lr": 1e-4,
            "epochs": 50
        },
        "resnet18_frozen": {
            "backbone": "resnet18", 
            "visual_dim": 256,
            "batch_size": 24,  # 冻结backbone可用更大batch size
            "lr": 1e-3,        # 冻结backbone可用更大学习率
            "epochs": 30,
            "freeze_backbone": True
        },
        "resnet34": {
            "backbone": "resnet34",
            "visual_dim": 256,
            "batch_size": 12,
            "lr": 5e-5,
            "epochs": 50
        },
        "resnet50": {
            "backbone": "resnet50",
            "visual_dim": 512,
            "batch_size": 8,   # ResNet50需要更小batch size
            "lr": 5e-5,        # ResNet50需要更小学习率
            "epochs": 40
        },
        "debug": {
            "backbone": "resnet18",
            "visual_dim": 256,
            "batch_size": 4,
            "lr": 1e-4,
            "epochs": 5,       # 调试用少量epochs
            "frame_interval": 10  # 调试用稀疏采样
        }
    }
    
    # 获取配置
    if config_name not in configs:
        print(f"Unknown config: {config_name}")
        print(f"Available configs: {list(configs.keys())}")
        return
    
    config = configs[config_name].copy()
    config.update(kwargs)  # 更新额外参数
    
    # 构建命令行参数
    for key, value in config.items():
        if key == "freeze_backbone" and value:
            cmd.append("--freeze_backbone")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running ResNet Stage2 training with config: {config_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Config: {config}")
    
    # 运行训练
    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
        print(f"\nTraining completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with return code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        return 1


def quick_test():
    """快速测试训练pipeline"""
    print("Running quick test...")
    return run_training("debug", epochs=2, batch_size=2)


def main():
    """主函数 - 提供交互式选择"""
    print("=" * 60)
    print("RESNET STAGE2 BEHAVIOR CLASSIFICATION")
    print("=" * 60)
    
    print("\nAvailable training configurations:")
    print("1. resnet18        - Standard ResNet18 (recommended)")
    print("2. resnet18_frozen - ResNet18 with frozen backbone (fast)")
    print("3. resnet34        - ResNet34 (better accuracy)")
    print("4. resnet50        - ResNet50 (best accuracy, slow)")
    print("5. debug           - Debug mode (quick test)")
    print("6. quick_test      - Very quick test")
    print("7. custom          - Custom parameters")
    
    choice = input("\nSelect configuration (1-7) or 'q' to quit: ").strip()
    
    if choice == 'q':
        print("Exiting...")
        return
    
    config_map = {
        '1': 'resnet18',
        '2': 'resnet18_frozen', 
        '3': 'resnet34',
        '4': 'resnet50',
        '5': 'debug',
        '6': None,  # Special case for quick_test
        '7': None   # Special case for custom
    }
    
    if choice == '6':
        quick_test()
        return
    
    if choice == '7':
        # Custom parameters
        print("\nCustom configuration:")
        backbone = input("Backbone (resnet18/resnet34/resnet50) [resnet18]: ").strip() or "resnet18"
        batch_size = int(input("Batch size [16]: ") or "16")
        epochs = int(input("Epochs [50]: ") or "50")
        lr = float(input("Learning rate [1e-4]: ") or "1e-4")
        
        freeze = input("Freeze backbone? (y/n) [n]: ").strip().lower() == 'y'
        
        custom_config = {
            'backbone': backbone,
            'batch_size': batch_size,
            'epochs': epochs, 
            'lr': lr
        }
        
        if freeze:
            custom_config['freeze_backbone'] = True
        
        return run_training("resnet18", **custom_config)
    
    elif choice in config_map and config_map[choice] is not None:
        config_name = config_map[choice]
        
        # 可选参数
        print(f"\nSelected: {config_name}")
        data_path = input("Data path [../dataset]: ").strip() or "../dataset"
        
        return run_training(config_name, data_path=data_path)
    
    else:
        print("Invalid choice!")
        return 1


if __name__ == '__main__':
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        if config_name == "test":
            quick_test()
        else:
            run_training(config_name)
    else:
        main()