# JRDB数据集格式与标注分析报告

## 数据集概述

### 基本信息
- **数据集名称**: JRDB (JackRabbot Dataset) - Social Activity & Pose 数据集
- **数据类型**: 多模态机器人社交数据集
- **场景数量**: 27个不同场景
- **总帧数**: 27,893张图像
- **数据格式**: 图像序列 + JSON标注文件
- **数据保存路径**: 图像序列：D:/1data/imagedata/images/image_stitched
- **数据保存路径**: JRDB—social标注文件：D:/1data/imagedata/labels/labels_2d_activity_social_stitched
- **数据保存路径**: 骨架JRDB-ACT标注文件：D:/1data/imagedata/labels/labels_2d_pose_stitched_coco

### 数据集结构
```
D:/1data/imagedata/
├── images/
│   └── image_stitched/              # 图像数据
│       ├── scene1/                  # 场景文件夹
│       │   ├── 000000.jpg          # 帧图像
│       │   ├── 000001.jpg
│       │   └── ...
│       └── scene2/
├── labels/
│   ├── labels_2d_activity_social_stitched/    # 社交活动标注
│   │   ├── scene1.json
│   │   └── scene2.json
│   └── labels_2d_pose_stitched_coco/          # 姿态标注
│       ├── scene1.json
│       └── scene2.json
```

## 社交活动标注格式 (Social Activity)

### 文件结构
```json
{
    "labels": {
        "000000.jpg": [
            {
                "action_label": {"sitting": 1, "talking to someone": 1},
                "attributes": {
                    "area": 42701.01456747742,
                    "interpolated": false,
                    "no_eval": false,
                    "occlusion": "Fully_visible",
                    "truncated": "false"
                },
                "box": [2962, 63, 299, 397],
                "file_id": "000002.jpg",
                "label_id": "pedestrian:6",
                "social_activity": {"sitting": 1.0},
                "social_group": {"cluster_ID": 7, "cluster_stat": 1},
                "demographics_info": [
                    {
                        "gender": {"Female": "1"},
                        "age": {"Young_Adulthood": "1"},
                        "race": {"Mongoloid/Asian": "1"}
                    }
                ],
                "H-interaction": [
                    {
                        "pair": "pedestrian:7",
                        "box_pair": [3359, 106, 236, 366],
                        "inter_labels": {"sitting together": 1}
                    }
                ],
                "group_info": [
                    {
                        "venue": {"indoor_cafeteria/dining_hall/food_court": "1"},
                        "aim": {"socializing&eating/ordering_food": "1"},
                        "inter": {"sitting": "1"},
                        "BPC": {"chair": "1"},
                        "SSC": {"table": "1"},
                        "location_pre": {"at": "1"}
                    }
                ]
            }
        ]
    }
}
```

### 标注字段详解

#### 基本信息
- **box**: [x, y, width, height] - 边界框坐标
- **label_id**: "pedestrian:X" - 人物唯一标识符
- **file_id**: 对应的图像文件名

#### 动作标签 (Action Labels)
- **action_label**: 人物当前执行的动作
  - 常见动作: sitting, walking, standing, talking, holding sth, looking at robot等

#### 社交互动 (H-interaction)
- **pair**: 互动对象的label_id
- **box_pair**: 互动对象的边界框
- **inter_labels**: 互动类型标签

#### 人口统计信息 (Demographics)
- **gender**: 性别标注 (Male/Female)
- **age**: 年龄组 (Young_Adulthood/Middle_Adulthood等)
- **race**: 种族标注

#### 场景信息 (Group Info)
- **venue**: 场所类型
- **aim**: 行为目的
- **BPC**: 身体接触物体 (Body Physical Contact)
- **SSC**: 空间语义接触 (Spatial Semantic Contact)

## 姿态标注格式 (Pose)

### COCO格式结构
```json
{
    "images": [
        {
            "id": 1,                    // 图像ID，用于关联annotations
            "width": 752,
            "height": 480,
            "file_name": "image_stitched/scene/000000.jpg"  // 完整图像路径
        }
    ],
    "annotations": [
        {
            "id": 1913,                 
            "track_id": 2,              
            "image_id": 2,              
            "category_id": 1,           // 类别ID (person=1)
            "keypoints": [x1, y1, v1, x2, y2, v2, ...],  // 17个关键点坐标和可见性
            "num_keypoints": 17         // 检测到的关键点数量
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "keypoints": ["head", "right eye", "left eye", ...]
        }
    ]
}
```

### 关键ID对应关系说明

#### 1. 图像文件对应关系
- **pose标注中**: `images[i].file_name` = "image_stitched/scene/000000.jpg"
- **实际文件路径**: `D:/1data/imagedata/images/image_stitched/scene/000000.jpg`
- **social标注中**: 直接使用文件名 "000000.jpg" 作为key

#### 2. 人物ID对应关系 (核心对应)
- **pose标注**: `annotations[i].track_id` = 2
- **social标注**: `label_id` = "pedestrian:2"
- **对应规则**: `track_id` == `pedestrian:X` 中的 X

#### 3. 帧图像ID对应关系
- **pose标注**: `images[i].id` ↔ `annotations[j].image_id`
- **作用**: 将姿态标注关联到具体的图像帧
- **注意**: 这是pose标注内部的ID系统，与social标注无关

## 数据融合：Pose + Social标注的ID匹配

### 完整的ID匹配流程

#### 步骤1: 图像帧匹配
```python
# 1. 从pose标注获取图像信息
pose_image = {
    "id": 1,
    "file_name": "image_stitched/meyer-green-2019-03-16_0/000000.jpg"
}

# 2. 提取文件名用于匹配social标注
frame_name = os.path.basename(pose_image["file_name"])  # "000000.jpg"

# 3. 在social标注中查找对应帧
social_frame_data = social_labels[frame_name]  # 返回该帧的所有人物标注
```

#### 步骤2: 人物ID匹配
```python
# 1. 从pose标注获取人物信息
pose_annotation = {
    "id": 1913,
    "track_id": 2,        # 关键：人物追踪ID
    "image_id": 1,
    "keypoints": [...]
}

# 2. 在social标注中查找对应人物
for person in social_frame_data:
    if person["label_id"] == "pedestrian:2":  # 匹配 track_id = 2
        # 找到对应人物，可以获取边界框和社交信息
        bbox = person["box"]
        interactions = person["H-interaction"]
        break
```

### 实际数据示例

#### Social标注示例 (bytes-cafe-2019-02-07_0.json)
```json
{
    "labels": {
        "000000.jpg": [
            {
                "label_id": "pedestrian:6",     // 人物ID = 6
                "box": [2962, 63, 299, 397],    // 边界框
                "action_label": {"sitting": 1},
                "H-interaction": [
                    {
                        "pair": "pedestrian:7",
                        "inter_labels": {"sitting together": 1}
                    }
                ]
            }
        ]
    }
}
```

#### 对应的Pose标注示例
```json
{
    "images": [
        {
            "id": 1,
            "file_name": "image_stitched/bytes-cafe-2019-02-07_0/000000.jpg"
        }
    ],
    "annotations": [
        {
            "id": 14616,
            "track_id": 6,          // 对应 pedestrian:6
            "image_id": 1,          // 对应 images.id = 1
            "keypoints": [2499.0, 3.5, 2, 2498.0, 43.0, 2, ...],
            "num_keypoints": 17
        }
    ]
}
```

### ID对应关系总结表

| 标注类型 | 文件匹配 | 人物匹配 | 边界框来源 | 备注 |
|---------|---------|----------|-----------|------|
| Social | 直接文件名 | `pedestrian:X` | `box`字段 | 包含社交互动信息 |
| Pose | 完整路径 | `track_id = X` | 无（需从Social获取） | 包含17个关键点 |
| **融合规则** | `basename(pose.file_name) == social.key` | `pose.track_id == social.pedestrian:X 中的 X` | 从Social标注获取 | 两种标注的桥梁 |

### 数据融合伪代码（基于实际实现）

```python
def get_frame_poses(pose_annotations, frame_name):
    """获取指定帧的pose数据 - 基于jrdbpose.py实现"""
    # 1. 查找对应的图像ID
    frame_id = None
    for img in pose_annotations['images']:
        # 提取文件名进行匹配（关键实现细节）
        img_filename = os.path.basename(img['file_name'])
        if img_filename == frame_name:
            frame_id = img['id']
            break
    
    if frame_id is None:
        return []
    
    # 2. 获取该帧的所有pose标注
    poses = []
    for ann in pose_annotations['annotations']:
        if ann['image_id'] == frame_id:
            poses.append(ann)
    
    return poses

def get_bbox_from_social(social_annotations, frame_name, person_id):
    """从social标注获取边界框 - 基于jrdbpose.py实现"""
    frame_data = social_annotations.get(frame_name, [])
    
    for person in frame_data:
        # 解析label_id获取人物ID
        label_parts = person['label_id'].split(':')
        if len(label_parts) > 1:
            social_id = int(label_parts[1])  # "pedestrian:6" -> 6
            if social_id == person_id:  # 匹配track_id
                return person['box']
    
    return None

def visualize_pose_with_social(scene_name, frame_name):
    """完整的可视化流程 - 基于实际代码逻辑"""
    # 1. 加载两种标注
    pose_annotations = load_pose_annotations(scene_name)
    social_annotations = load_social_annotations(scene_name)
    
    # 2. 获取该帧的pose数据
    frame_poses = get_frame_poses(pose_annotations, frame_name)
    
    # 3. 为每个pose获取对应的社交信息
    for pose_data in frame_poses:
        track_id = pose_data.get('track_id', 0)
        keypoints = pose_data['keypoints']
        
        # 4. 通过track_id获取边界框（关键对应关系）
        bbox = get_bbox_from_social(social_annotations, frame_name, track_id)
        
        # 5. 可选：获取社交互动信息
        social_info = None
        frame_data = social_annotations.get(frame_name, [])
        for person in frame_data:
            if person['label_id'] == f"pedestrian:{track_id}":
                social_info = {
                    'actions': person.get('action_label', {}),
                    'interactions': person.get('H-interaction', [])
                }
                break
```

### 实际代码中的关键实现细节

#### jrdbpose.py中的ID匹配逻辑
```python
# 文件名匹配：完整路径 -> 文件名
img_filename = os.path.basename(img['file_name'])  # 提取000000.jpg

# 人物ID匹配：track_id -> pedestrian:X
social_id = int(label_parts[1])  # "pedestrian:6" -> 6
if social_id == person_id:  # person_id来自pose的track_id
```

#### jrdbsocial.py中的互动处理
```python
# 社交互动可视化
for person in frame_annotations:
    current_id = person['label_id']  # "pedestrian:6"
    for interaction in person['H-interaction']:
        pair_id = interaction['pair']  # "pedestrian:7" 
        pair_box = interaction['box_pair']  # 对方的边界框
        inter_labels = interaction['inter_labels']  # 互动类型
```

### 代码实现验证

基于实际代码检查，确认以下实现细节：

#### ✅ 已验证的一致性
1. **文件名匹配**: `os.path.basename(img['file_name']) == frame_name` ✓
2. **人物ID解析**: `int(person['label_id'].split(':')[1]) == track_id` ✓
3. **边界框获取**: 从social标注的`box`字段获取 ✓
4. **互动关系**: 使用`H-interaction.pair`和`box_pair` ✓

#### ⚠️ 实现注意事项
1. **错误处理**: 代码中包含`person_id = pose_data.get('track_id', 0)`的默认值处理
2. **文件路径**: pose中存储完整路径，需要`basename()`提取文件名
3. **数据结构**: social标注使用`data['labels'][frame]`层次结构
4. **ID匹配**: 通过字符串解析`pedestrian:X`获取数字ID

### 关键对应关系总结（基于代码验证）

| 步骤 | Pose标注 | Social标注 | 实现代码 |
|------|---------|-----------|----------|
| 1. 文件匹配 | `basename(file_name)` | 直接key | `os.path.basename()` |
| 2. 图像ID | `images.id → annotations.image_id` | N/A | 循环查找匹配 |
| 3. 人物匹配 | `track_id` | `split(':')[1]` | 字符串解析 |
| 4. 边界框 | 无 | `box` | `get_bbox_from_social()` |

### 注意事项

1. **ID不连续性**: 并非所有帧都有pose标注，且track_id可能不连续
2. **标注缺失**: 某些人物可能只有social标注而无pose标注，反之亦然
3. **时序一致性**: track_id在时间序列中保持一致，用于跨帧追踪
4. **边界框来源**: Pose标注中没有bbox信息，必须从Social标注获取
5. **坐标系一致性**: 两种标注使用相同的图像坐标系
6. **错误容忍**: 代码包含了找不到匹配时的默认处理（返回None或0）

### 关键点定义 (17个关键点)
1. head (0)
2. right eye (1)
3. left eye (2)
4. right shoulder (3)
5. center shoulder (4)
6. left shoulder (5)
7. right elbow (6)
8. left elbow (7)
9. center hip (8)
10. right wrist (9)
11. right hip (10)
12. left hip (11)
13. left wrist (12)
14. right knee (13)
15. left knee (14)
16. right foot (15)
17. left foot (16)

### 骨架连接
```python
SKELETON_CONNECTIONS = [
    (1, 2),   # right eye - left eye
    (0, 4),   # head - center shoulder
    (3, 4),   # right shoulder - center shoulder
    (8, 10),  # center hip - right hip
    (5, 7),   # left shoulder - left elbow
    (10, 13), # right hip - right knee
    (14, 16), # left knee - left foot
    (4, 5),   # center shoulder - left shoulder
    (7, 12),  # left elbow - left wrist
    (4, 8),   # center shoulder - center hip
    (3, 6),   # right shoulder - right elbow
    (13, 15), # right knee - right foot
    (8, 11),  # center hip - left hip
    (6, 9),   # right elbow - right wrist
    (11, 14)  # left hip - left knee
]
```

## H-Interaction统计分析

### 总体统计
- **处理文件数**: 27个场景
- **检测人物总数**: 843,133人
- **互动总数**: 1,452,876次
- **互动类型总数**: 19种

### 互动类型分布

| 排名 | 互动类型 | 出现次数 | 占比 | 描述 |
|------|----------|----------|------|------|
| 1 | walking together | 681,872 | 46.9% | 一起行走 |
| 2 | standing together | 489,553 | 33.7% | 一起站立 |
| 3 | conversation | 128,474 | 8.8% | 对话交流 |
| 4 | sitting together | 112,207 | 7.7% | 一起坐着 |
| 5 | going upstairs together | 9,274 | 0.6% | 一起上楼 |
| 6 | moving together | 8,154 | 0.6% | 一起移动 |
| 7 | looking at robot together | 7,732 | 0.5% | 一起看机器人 |
| 8 | looking at sth together | 3,640 | 0.3% | 一起看某物 |
| 9 | walking toward each other | 3,476 | 0.2% | 相向而行 |
| 10 | eating together | 3,452 | 0.2% | 一起进食 |
| 11 | going downstairs together | 1,434 | 0.1% | 一起下楼 |
| 12 | bending together | 1,080 | 0.1% | 一起弯腰 |
| 13 | holding sth together | 1,080 | 0.1% | 一起拿东西 |
| 14 | cycling together | 1,076 | 0.1% | 一起骑车 |
| 15 | interaction with door together | 122 | 0.0% | 一起与门互动 |
| 16 | hugging | 104 | 0.0% | 拥抱 |
| 17 | looking into sth together | 58 | 0.0% | 一起查看 |
| 18 | shaking hand | 48 | 0.0% | 握手 |
| 19 | waving hand together | 40 | 0.0% | 一起挥手 |

### 互动类型分类

#### 移动类互动 (80.6%)
- walking together (46.9%)
- standing together (33.7%)

#### 交流类互动 (8.8%)
- conversation (8.8%)

#### 共同活动类互动 (8.5%)
- sitting together (7.7%)
- eating together (0.2%)
- looking at robot together (0.5%)
- 其他 (0.1%)

#### 物理接触类互动 (0.1%)
- hugging (0.0%)
- shaking hand (0.0%)

## 数据标注的关键特点

### 1. 多模态标注
- 同时包含社交活动和人体姿态信息
- 丰富的人口统计学标注
- 详细的场景上下文信息

### 2. 时序一致性
- 连续帧之间的人物追踪
- 插值标注处理缺失帧

### 3. 层次化标注
- 个体行为 → 双人互动 → 群体活动
- 动作 → 意图 → 场景上下文

### 4. 标注质量指标
- 遮挡程度标注 (occlusion)
- 截断标注 (truncated)
- 插值标注标记 (interpolated)

## 训练应用建议

### 1. 社交行为识别
- **输入**: 图像序列 + 边界框
- **输出**: 19种H-interaction分类
- **特点**: 不平衡数据分布，需要重采样策略
- **ID使用**: 使用social标注的`pedestrian:X`和`H-interaction.pair`建立人物关系图

### 2. 人体姿态估计
- **输入**: RGB图像
- **输出**: 17个关键点坐标
- **特点**: COCO格式兼容，可迁移学习
- **ID使用**: 通过`track_id`匹配social标注获取边界框，提供ROI输入

### 3. 多人追踪
- **输入**: 视频序列
- **输出**: 跨帧人物ID对应
- **特点**: track_id提供监督信号
- **ID使用**: `track_id`作为ground truth，训练ReID模型

### 4. 场景理解
- **输入**: 图像 + 人物标注
- **输出**: 场所类型、行为目的分类
- **特点**: 丰富的上下文标注
- **ID使用**: 通过人物ID聚合群体行为，分析场景语义

### 5. 多模态融合任务（推荐）
- **输入**: RGB图像 + 历史帧序列
- **输出**: 人物动作 + 姿态 + 社交互动
- **融合策略**: 
  ```python
  # 示例融合流程
  for frame in video_sequence:
      # 1. 提取pose特征
      pose_features = extract_pose_features(frame.keypoints)
      
      # 2. 提取社交特征  
      social_features = extract_social_features(frame.interactions)
      
      # 3. 通过track_id关联多人信息
      multi_person_context = aggregate_by_track_id(pose_features, social_features)
      
      # 4. 时序建模
      sequence_features = temporal_modeling(multi_person_context, history_frames)
  ```

## 数据预处理建议

### 1. 数据平衡
- 对低频互动类型进行上采样
- 使用加权损失函数处理类别不平衡

### 2. 数据增强
- 空间变换：旋转、缩放、翻转
- 时序增强：帧采样、速度变化
- 遮挡模拟：随机遮挡部分区域

### 3. 标注处理
- 处理插值标注的不确定性
- 统一坐标系和归一化
- 缺失标注的填充策略

### 4. 多模态融合
- RGB图像特征
- 人体姿态特征
- 时序运动特征
- 场景上下文特征

## 评估指标建议

### 分类任务
- 准确率 (Accuracy)
- F1-Score (考虑类别不平衡)
- 混淆矩阵分析

### 检测任务
- mAP (mean Average Precision)
- IoU阈值下的精度召回率

### 时序任务
- 帧级准确率
- 序列级准确率
- 时序一致性指标

---

*本报告基于对27个JRDB场景的完整分析，为后续的深度学习模型训练提供数据集格式参考和训练策略建议。*