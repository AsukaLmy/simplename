# Stage2 三分类行为检测系统

本文档说明如何使用修改后的Stage2系统，该系统专门针对三种基础行为类别进行分类。

## 修改概述

### 问题解决
- **原始问题**: 同一交互对可能有多个交互行为标签，导致训练不稳定
- **解决方案**: 简化为3个互斥的基础行为类别，过滤掉其他所有交互类型

### 新的3分类系统
1. **Walking Together** (0) - 一起走路
2. **Standing Together** (1) - 一起站立  
3. **Sitting Together** (2) - 一起坐着

### 过滤掉的交互类型
所有其他交互类型将被**完全过滤**，不参与训练：
- `conversation`
- `hugging`
- `eating together`
- `going upstairs together`
- 等其他15种交互类型

## 使用方法

### 1. 训练Stage2三分类模型

```bash
cd GeometricDete
python train_geometric_stage2.py \
    --data_path /path/to/your/dataset \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --use_attention \
    --history_length 5 \
    --early_stopping_patience 15 \
    --early_stopping_metric mpca
```

### 2. 关键参数说明

#### 数据处理参数
- `--history_length 5`: 时序历史长度
- `--use_temporal`: 启用时序特征（可选）
- `--use_scene_context`: 启用场景上下文特征

#### 模型参数
- `--hidden_dims 64 32 16`: 隐藏层维度
- `--dropout 0.2`: Dropout率
- `--use_attention`: 启用注意力机制

#### 训练参数
- `--mpca_weight 0.03`: MPCA正则化权重（平衡各类准确率）
- `--acc_weight 0.01`: 准确率正则化权重
- `--max_grad_norm 1.0`: 梯度裁剪

#### 类别平衡
系统自动应用类别权重：
```python
class_weights = {
    0: 1.0,    # Walking Together - 基准
    1: 1.4,    # Standing Together - 轻微提升
    2: 6.1     # Sitting Together - 大幅提升（原本最少）
}
```

### 3. 测试修改

运行测试脚本验证系统：
```bash
python test_3class_dataset.py
```

## 核心修改文件

### 1. `geometric_stage2_dataset.py`
- 修改了`Stage2LabelMapper`类
- 添加了`is_valid_interaction()`方法
- 更新了`_apply_stage2_labels()`方法以过滤无效样本
- 调整了过采样逻辑以适应3分类

### 2. `geometric_stage2_classifier.py` 
- 将输出维度从4改为3
- 更新了损失函数以适应3分类
- 修改了评估器的类别数量

### 3. `train_geometric_stage2.py`
- 更新了类别权重设置
- 修改了类别名称
- 调整了报告生成逻辑

## 预期效果

### 数据过滤
- 只保留有效的3种基础行为类别
- 自动过滤掉有标签冲突的交互对
- 大幅减少数据集大小，但提高数据质量

### 训练稳定性
- 消除了多标签冲突问题
- 简化了分类任务
- 提高了模型训练的稳定性

### 性能指标
- **准确率**: 整体分类准确率
- **MPCA**: 各类平均准确率（推荐作为主要指标）
- **F1-Score**: 宏平均和加权平均F1分数

## 输出文件

训练完成后会生成：
- `stage2_experiments/stage2_geometric_stage2_YYYYMMDD_HHMMSS/`
  - `best_model.pth`: 最佳模型权重
  - `config.json`: 训练配置
  - `evaluation_report.txt`: 详细评估报告
  - `training_curves.png`: 训练曲线图
  - `test_evaluation_report.txt`: 测试集评估报告（如有）

## 注意事项

1. **数据要求**: 确保数据集中包含足够的3种基础行为类别样本
2. **类别不平衡**: 系统会自动处理类别不平衡问题
3. **特征维度**: 输入特征必须是16维（7几何+4运动+5时序）
4. **早停策略**: 建议使用MPCA作为早停指标，更好地平衡各类性能

## 示例使用

```python
# 创建数据加载器
from optimized_stage2_data_loader import create_fast_stage2_data_loaders

train_loader, val_loader, test_loader = create_fast_stage2_data_loaders(
    data_path="path/to/data",
    batch_size=64,
    num_workers=2,
    history_length=5,
    use_temporal=False,
    use_scene_context=True
)

# 创建模型
from geometric_stage2_classifier import GeometricStage2Classifier

model = GeometricStage2Classifier(
    input_dim=16,
    hidden_dims=[64, 32, 16], 
    dropout=0.2,
    use_attention=True
)

# 开始训练
from train_geometric_stage2 import GeometricStage2Trainer

trainer = GeometricStage2Trainer(config)
trainer.train(train_loader, val_loader, test_loader)
```

这个修改确保了Stage2只处理清晰、互斥的基础行为类别，避免了多标签冲突问题。