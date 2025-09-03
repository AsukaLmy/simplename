# DIN模型应用于JRDB数据集项目实施计划

## 项目概述

将现有的DIN (Dynamic Inference Network) 群体活动识别模型适配至JRDB (JackRabbot Dataset)数据集，实现对复杂社交场景中人物行为和群体活动的识别。

### 目标
- 将DIN模型从Volleyball/Collective数据集扩展到JRDB数据集
- 利用JRDB丰富的社交互动标注实现更细粒度的群体活动识别
- 保持原有二阶段训练流程：Stage1(个体动作识别) → Stage2(群体活动识别)
- 适配推理模块支持JRDB数据格式

## 数据集分析

### JRDB数据集特点
- **数据规模**: 27个场景，27,893张图像
- **标注类型**: 
  - 社交活动标注：19种H-interaction类型
  - 人体姿态标注：17个关键点（COCO格式）
- **数据结构**:
  ```
  D:/1data/imagedata/
  ├── images/image_stitched/scene*/000000.jpg
  ├── labels/labels_2d_activity_social_stitched/scene*.json
  └── labels/labels_2d_pose_stitched_coco/scene*.json
  ```

### 与现有数据集对比
| 特征 | Volleyball | Collective | JRDB |
|------|-----------|-----------|------|
| 活动类型 | 8种群体活动 | 5种群体活动 | 19种社交互动 |
| 标注粒度 | 群体+个体动作 | 群体+个体动作 | 社交互动+姿态 |
| 场景复杂度 | 单一运动场景 | 多样化场景 | 复杂社交场景 |

## 技术实施方案

### 1. 数据适配层设计

#### 1.1 创建JRDB数据集类
创建 `jrdb.py` 文件，实现：
- `JRDBDataset` 类继承自PyTorch Dataset
- `jrdb_read_dataset()` 函数读取标注文件
- `jrdb_all_frames()` 函数提取所有训练帧
- 数据加载器适配函数

#### 1.2 标注格式转换
- **ID映射**: `pose.track_id` ↔ `social.pedestrian:X`
- **边界框获取**: 从social标注提取bbox用于ROI Align
- **活动标签映射**: 将19种H-interaction映射为数值标签
- **姿态特征**: 利用17个关键点作为额外特征

#### 1.3 配置参数适配
修改 `config.py` 添加JRDB配置：
```python
elif dataset_name=='jrdb':
    self.data_path = 'D:/1data/imagedata'
    self.num_activities = 19  # H-interaction类型数量
    self.num_actions = 9     # 保持个体动作分类数
    # 场景分割配置
    self.train_scenes = [...]  # 训练场景列表
    self.test_scenes = [...]   # 测试场景列表
```

### 2. 训练脚本开发

#### 2.1 train_jrdb_stage1.py
基于 `train_collective_stage1.py` 修改：
- **输入**: RGB图像 + 边界框 + 姿态关键点
- **输出**: 个体动作分类
- **损失函数**: 交叉熵损失
- **数据增强**: 适配JRDB图像尺寸

#### 2.2 train_jrdb_stage2.py  
基于 `train_collective_stage2.py` 修改：
- **输入**: Stage1特征 + 空间关系图
- **输出**: 5种H-interaction分类（重新映射后）
- **网络结构**: 加载Stage1预训练模型，训练GCN层
- **损失函数**: 基于频率权重的交叉熵损失

### 3. 模型架构适配

#### 3.1 基础网络适配
修改 `base_model.py`:
- **输入通道**: 适配JRDB图像格式
- **ROI Align**: 利用social标注的bbox
- **特征融合**: RGB特征 + 姿态特征，要求提供多种backbone选项（默认mobilenetv2）

#### 3.2 GCN网络扩展
修改 `gcn_model.py`:
- **节点特征**: 个体动作特征 + 姿态特征
- **边权重**: 空间距离 + 社交关系权重
- **输出层**: 适配5类H-interaction（重新映射后）

### 4. 推理模块适配

#### 4.1 动态推理模块
修改 `infer_module/dynamic_infer_module.py`:
- **输入处理**: 支持JRDB数据格式
- **特征提取**: 集成姿态信息
- **后处理**: H-interaction结果输出

#### 4.2 可视化功能
- 绘制人物边界框
- 显示检测到的关键点
- 标注预测的社交互动类型
- 时序一致性分析



## 数据加载指南

### H-interaction标签映射（解决类别不平衡）

为解决JRDB数据集中严重的类别不平衡问题，将19种H-interaction类型重新映射为5类：

#### 标签映射表
| 原始类别 | 出现频率 | 新标签 | 新类别名 |
|---------|---------|-------|---------|
| walking together | 46.9% | 0 | walking_together |
| standing together | 33.7% | 1 | standing_together |
| conversation | 8.8% | 2 | conversation |
| sitting together | 7.7% | 3 | sitting_together |
| **其他15种类型** | 2.9% | 4 | others |

#### "others"类别包含的原始标签：
- going upstairs together (0.6%)
- moving together (0.6%) 
- looking at robot together (0.5%)
- looking at sth together (0.3%)
- walking toward each other (0.2%)
- eating together (0.2%)
- going downstairs together (0.1%)
- bending together (0.1%)
- holding sth together (0.1%)
- cycling together (0.1%)
- interaction with door together (0.0%)
- hugging (0.0%)
- looking into sth together (0.0%)
- shaking hand (0.0%)
- waving hand together (0.0%)

#### 实现代码示例
```python
def map_interaction_labels(original_label):
    """映射H-interaction标签到5类系统"""
    main_interactions = {
        'walking together': 0,
        'standing together': 1, 
        'conversation': 2,
        'sitting together': 3
    }
    
    return main_interactions.get(original_label, 4)  # 其他类别映射为4
```

#### 配置更新
```python
elif dataset_name=='jrdb':
    self.data_path = 'D:/1data/imagedata'
    self.num_activities = 5  # 重新映射后的类别数量
    self.num_actions = 9     # 保持个体动作分类数
    
    # 类别权重（解决不平衡）
    self.activity_weights = [1.0, 1.4, 5.3, 6.1, 16.7]  # 基于频率倒数
```

## 关键技术挑战

### 1. 数据不平衡问题
- **挑战**: 即使重新映射后，walking together仍占46.9% vs others仅2.9%
- **解决方案**: 
  - 基于频率的加权损失函数
  - focal loss处理困难样本
  - 对低频类别进行上采样

### 2. 多模态特征融合
- **挑战**: RGB特征 + 姿态特征有效融合
- **解决方案**:
  - 注意力机制
  - 特征对齐
  - 多尺度融合

### 3. 时序建模
- **挑战**: 社交互动的时序依赖性
- **解决方案**:
  - LSTM/GRU时序建模
  - 滑动窗口采样
  - 时序注意力

## 评估指标

### 分类性能
- **准确率 (Accuracy)**
- **F1-Score** (处理类别不平衡)  
- **混淆矩阵分析**
- **每类精确率/召回率**

### 检测性能
- **mAP** (mean Average Precision)
- **IoU阈值下的性能**

### 时序性能
- **帧级准确率**
- **序列级准确率** 
- **时序一致性指标**

## 预期产出

### 1. 代码文件
```
jrdbimply
├── jrdb.py                    # JRDB数据集类
├── scripts/
│   ├── train_jrdb_stage1.py   # Stage1训练脚本
│   └── train_jrdb_stage2.py   # Stage2训练脚本
├── infer_module/
│   └── dynamic_infer_jrdb.py  # JRDB推理模块
└── config.py                  # 更新配置文件
```

### 2. 预训练模型
- `jrdb_stage1_best.pth` - Stage1最佳模型
- `jrdb_stage2_best.pth` - Stage2最佳模型

### 3. 实验结果
- 训练日志和学习曲线
- 测试集性能报告
- 可视化结果样例
- 消融实验分析

## 扩展方向

### 1. 多尺度时序建模
- 短期行为识别 (1-3秒)
- 长期活动理解 (10-30秒)

### 2. 场景上下文建模
- 环境语义理解
- 物体交互关系

### 3. 多模态扩展
- 音频信息融合
- 深度信息利用

---

*本计划基于DIN模型架构和JRDB数据集特点制定，旨在实现高效的社交场景群体活动识别系统。*