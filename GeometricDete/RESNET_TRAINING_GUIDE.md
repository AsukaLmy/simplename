# ResNet-based Stage2 Behavior Classification Training Guide

## 🚀 Quick Start

### 1. 简单启动（推荐）
```bash
# 交互式选择配置
python run_resnet_training.py

# 直接运行ResNet18配置
python run_resnet_training.py resnet18

# 快速测试
python run_resnet_training.py test
```

### 2. 命令行启动
```bash
# ResNet18 标准配置
python train_resnet_stage2.py --backbone resnet18 --visual_dim 256 --batch_size 16 --epochs 50

# ResNet18 冻结backbone（更快）
python train_resnet_stage2.py --backbone resnet18 --freeze_backbone --batch_size 24 --lr 1e-3

# ResNet50 高精度配置
python train_resnet_stage2.py --backbone resnet50 --visual_dim 512 --batch_size 8 --lr 5e-5 --epochs 40
```

## 📋 配置选项

### 预定义配置

| 配置名 | Backbone | 特征维度 | Batch Size | 学习率 | 特点 |
|--------|----------|----------|------------|--------|------|
| resnet18 | ResNet18 | 256 | 16 | 1e-4 | 标准配置，推荐 |
| resnet18_frozen | ResNet18 | 256 | 24 | 1e-3 | 冻结backbone，训练快 |
| resnet34 | ResNet34 | 256 | 12 | 5e-5 | 更好精度 |
| resnet50 | ResNet50 | 512 | 8 | 5e-5 | 最佳精度，需要更多GPU内存 |
| debug | ResNet18 | 256 | 4 | 1e-4 | 调试用，少量数据 |

### 命令行参数

#### 模型参数
- `--backbone`: ResNet架构 (resnet18/resnet34/resnet50)
- `--visual_dim`: 视觉特征维度 (默认: 256)
- `--fusion`: 特征融合策略 (concat/bilinear/add)
- `--freeze_backbone`: 冻结ResNet backbone参数

#### 训练参数
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批大小 (默认: 16)
- `--lr`: 学习率 (默认: 1e-4)
- `--weight_decay`: 权重衰减 (默认: 1e-5)

#### 数据参数
- `--data_path`: 数据集路径 (默认: ../dataset)
- `--frame_interval`: 帧采样间隔 (默认: 1)
- `--num_workers`: 数据加载线程数 (默认: 4)

#### 其他参数
- `--checkpoint_dir`: 检查点保存目录
- `--device`: 设备 (auto/cpu/cuda)
- `--log_interval`: 日志打印间隔
- `--seed`: 随机种子

## 📊 训练监控

### 输出日志示例
```
==============================================================
RESNET STAGE2 CONFIGURATION
==============================================================
Model Architecture:
  Type: resnet_relation
  Backbone: resnet18
  Visual features: 256D
  Spatial features: 8D
  Fusion strategy: concat

Training:
  Epochs: 50
  Batch size: 16
  Learning rate: 0.0001
  Optimizer: adam

✅ ResNet Stage2 data loaders created:
   Train: 1,234 samples, 78 batches
   Val:   456 samples, 29 batches
   Test:  234 samples, 15 batches

Train Epoch: 1 [    0/1234 (  0%)] Loss: 1.098765
Train Epoch: 1 [  160/1234 ( 13%)] Loss: 1.045321
...
Val Epoch 1: Avg Loss: 0.987654, Acc: 0.4567, MPCA: 0.3890
New best model saved! mpca: 0.3890
```

### 关键指标
- **Accuracy**: 整体准确率
- **MPCA**: 平均每类准确率（主要优化目标）
- **Loss**: 总损失（CE + MPCA + Acc正则化）

## 🎯 预期性能

### 性能基准（相比HoG特征）

| 模型 | 预期验证准确率 | 预期MPCA | 训练时间 | GPU内存 |
|------|----------------|----------|----------|---------|
| ResNet18 | 65-75% | 0.60-0.70 | ~2-3小时 | ~4GB |
| ResNet18 (frozen) | 60-70% | 0.55-0.65 | ~1-2小时 | ~3GB |
| ResNet34 | 70-78% | 0.65-0.75 | ~3-4小时 | ~5GB |
| ResNet50 | 75-82% | 0.70-0.80 | ~4-6小时 | ~8GB |

### 与原有HoG方法对比
- **HoG特征**: 验证准确率 ~33% (随机水平)
- **ResNet特征**: 验证准确率 >65% (显著提升)
- **零值特征**: 从100% → ~55% (大幅改善)

## 🛠️ 故障排除

### 常见问题

#### 1. GPU内存不足
```bash
# 解决方案：减小batch size
python train_resnet_stage2.py --batch_size 8

# 或冻结backbone
python train_resnet_stage2.py --freeze_backbone --batch_size 16
```

#### 2. 训练过慢
```bash
# 解决方案：使用冻结backbone
python train_resnet_stage2.py --freeze_backbone --lr 1e-3

# 或增加帧间隔
python train_resnet_stage2.py --frame_interval 5
```

#### 3. 数据加载错误
```bash
# 检查数据路径
python train_resnet_stage2.py --data_path /path/to/your/dataset

# 减少数据加载线程
python train_resnet_stage2.py --num_workers 2
```

#### 4. 验证准确率不提升
- 检查学习率：可能过高或过低
- 检查数据质量：确保数据路径正确
- 增加训练轮数：ResNet可能需要更多轮数收敛

### 调试模式
```bash
# 快速调试（少量数据和轮数）
python run_resnet_training.py debug

# 更快的测试
python run_resnet_training.py test
```

## 📁 输出文件

训练完成后，检查点目录包含：
- `best_model.pth`: 最佳模型权重
- `final_results.json`: 最终测试结果
- 训练日志和指标

### 结果文件示例
```json
{
  "test_accuracy": 0.7234,
  "test_mpca": 0.6890,
  "best_val_accuracy": 0.7156, 
  "best_val_mpca": 0.6745,
  "config": {
    "backbone_name": "resnet18",
    "visual_feature_dim": 256,
    "fusion_strategy": "concat",
    "learning_rate": 0.0001,
    "batch_size": 16
  }
}
```

## 🔧 自定义配置

### 创建自定义配置
```python
from configs.resnet_stage2_config import ResNetStage2Config

custom_config = ResNetStage2Config(
    backbone_name='resnet34',
    visual_feature_dim=512,
    fusion_strategy='bilinear',
    batch_size=12,
    learning_rate=5e-5,
    epochs=40
)
```

### 高级特征融合策略
- `concat`: 特征拼接（默认，稳定）
- `bilinear`: 双线性融合（更复杂的交互）
- `add`: 元素级相加（需要相同维度）

## 📈 性能优化建议

1. **首次训练**: 使用 `resnet18` 标准配置
2. **快速实验**: 使用 `resnet18_frozen`
3. **追求精度**: 使用 `resnet50` 配置
4. **GPU内存限制**: 减小 `batch_size` 或使用 `freeze_backbone`
5. **训练时间限制**: 增大 `frame_interval` 或减少 `epochs`

## 🎉 预期改进

使用ResNet backbone后，你应该看到：
- ✅ 验证准确率从33%提升到65%+
- ✅ MPCA从随机水平提升到0.6+
- ✅ 特征质量显著改善（零值比例降低）
- ✅ 更好的泛化性能