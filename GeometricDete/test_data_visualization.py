#!/usr/bin/env python3
"""
数据载入过程可视化测试脚本
检查Stage2数据集的特征提取是否正确
"""

import os
import sys

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.stage2_dataset import BasicStage2Dataset
from configs.stage2_config import Stage2Config


def visualize_interaction_pair(sample, dataset, idx, save_dir="./visualization_output"):
    """
    可视化单个交互对
    
    Args:
        sample: 数据集样本
        dataset: 数据集对象
        idx: 样本索引
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取信息
    features = sample['features'].numpy()
    stage2_label = sample['stage2_label'].item()
    frame_id = sample['frame_id']
    original_interaction = sample['original_interaction']
    person_A_id = sample['person_A_id']
    person_B_id = sample['person_B_id']
    
    # 从数据集内部获取更多信息
    raw_sample = dataset.samples[idx]
    person_A_box = raw_sample['person_A_box']  # [x, y, w, h]
    person_B_box = raw_sample['person_B_box']  # [x, y, w, h]
    scene_name = raw_sample['scene_name']
    image_name = raw_sample['image_name']
    
    # 标签映射
    label_names = ['Walking Together', 'Standing Together', 'Sitting Together']
    label_name = label_names[stage2_label] if stage2_label < len(label_names) else f'Unknown_{stage2_label}'
    
    # 特征分解 (假设是72维：7几何+64HoG+1场景上下文)
    feature_dim = len(features)
    if feature_dim >= 72:
        geometric_features = features[:7]
        hog_features = features[7:71]
        scene_context = features[71]
    elif feature_dim >= 8:
        geometric_features = features[:7]
        hog_features = features[7:-1] if feature_dim > 8 else features[7:8]
        scene_context = features[-1]
    else:
        geometric_features = features
        hog_features = np.array([])
        scene_context = 0.0
    
    print(f"\n📊 交互对信息:")
    print(f"  场景: {scene_name}")
    print(f"  图像: {image_name}")
    print(f"  帧ID: {frame_id}")
    print(f"  人员A ID: {person_A_id}, 人员B ID: {person_B_id}")
    print(f"  原始交互: {original_interaction}")
    print(f"  标签: {label_name} (ID: {stage2_label})")
    print(f"  特征维度: {feature_dim}")
    
    print(f"\n🔢 特征统计:")
    if len(geometric_features) > 0:
        print(f"  几何特征 ({len(geometric_features)}维): mean={geometric_features.mean():.4f}, std={geometric_features.std():.4f}")
        if len(geometric_features) >= 7:
            print(f"    - 水平间距(归一化): {geometric_features[0]:.4f}")
            print(f"    - 高度比: {geometric_features[1]:.4f}") 
            print(f"    - 地面距离(归一化): {geometric_features[2]:.4f}")
            print(f"    - 垂直重叠: {geometric_features[3]:.4f}")
            print(f"    - 面积比: {geometric_features[4]:.4f}")
            print(f"    - 中心距离(归一化): {geometric_features[5]:.4f}")
            print(f"    - 垂直距离比: {geometric_features[6]:.4f}")
    
    if len(hog_features) > 0:
        print(f"  HoG特征 ({len(hog_features)}维): mean={hog_features.mean():.4f}, std={hog_features.std():.4f}")
    
    print(f"  场景上下文: {scene_context:.4f}")
    
    # 边界框信息
    print(f"\n📦 边界框信息:")
    print(f"  人员A: {person_A_box} (x,y,w,h)")
    print(f"  人员B: {person_B_box} (x,y,w,h)")
    
    # 全景图像边界检查
    center_A_x = person_A_box[0] + person_A_box[2] / 2
    center_B_x = person_B_box[0] + person_B_box[2] / 2
    dx_direct = abs(center_A_x - center_B_x)
    dx_wraparound = 3760 - dx_direct
    print(f"  水平距离: 直接={dx_direct:.1f}, 环绕={dx_wraparound:.1f}, 实际使用={'环绕' if dx_wraparound < dx_direct else '直接'}")
    
    # 检查是否可能跨越边界
    if (center_A_x < 500 and center_B_x > 3260) or (center_A_x > 3260 and center_B_x < 500):
        print(f"  ⚠️  检测到全景边界跨越情况！")
    
    # 尝试获取图像路径并可视化
    scene_info = dataset.scene_data.get(frame_id, {})
    image_path = scene_info.get('image_path')
    
    if image_path and os.path.exists(image_path):
        visualize_on_image(sample, raw_sample, image_path, save_dir)
    else:
        print(f"⚠️  图像文件不存在或路径无效: {image_path}")
    
    return sample


def visualize_on_image(sample, raw_sample, image_path, save_dir):
    """
    在原图上可视化交互对
    
    Args:
        sample: 处理后的样本
        raw_sample: 原始样本数据 
        image_path: 图像路径
        save_dir: 保存目录
    """
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图像副本用于绘制
        img_with_annotations = image.copy()
        
        # 设置matplotlib
        plt.figure(figsize=(15, 10))
        plt.imshow(img_with_annotations)
        
        # 获取边界框信息 (格式: [x, y, w, h])
        person_A_box = raw_sample['person_A_box']
        person_B_box = raw_sample['person_B_box']
        
        # 绘制Person A边界框 (蓝色)
        rect1 = Rectangle((person_A_box[0], person_A_box[1]), 
                         person_A_box[2], person_A_box[3],
                         linewidth=3, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect1)
        plt.text(person_A_box[0], person_A_box[1] - 10, 
                f'Person A (ID: {raw_sample["person_A_id"]})', 
                color='blue', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 绘制Person B边界框 (红色)
        rect2 = Rectangle((person_B_box[0], person_B_box[1]), 
                         person_B_box[2], person_B_box[3],
                         linewidth=3, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect2)
        plt.text(person_B_box[0], person_B_box[1] - 10, 
                f'Person B (ID: {raw_sample["person_B_id"]})', 
                color='red', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 绘制连接线
        center1 = (person_A_box[0] + person_A_box[2] // 2, 
                  person_A_box[1] + person_A_box[3] // 2)
        center2 = (person_B_box[0] + person_B_box[2] // 2, 
                  person_B_box[1] + person_B_box[3] // 2)
        
        plt.plot([center1[0], center2[0]], [center1[1], center2[1]], 
                'g--', linewidth=3, alpha=0.8, label='交互连线')
        
        # 添加标题和信息
        stage2_label = sample['stage2_label'].item()
        label_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        label_name = label_names[stage2_label] if stage2_label < len(label_names) else f'Unknown_{stage2_label}'
        
        plt.title(f'Stage2交互对可视化\n'
                 f'场景: {raw_sample["scene_name"]} | 图像: {raw_sample["image_name"]}\n'
                 f'原始交互: {sample["original_interaction"]} | 标签: {label_name} (ID: {stage2_label})', 
                 fontsize=14, pad=20)
        
        plt.axis('off')
        plt.legend(loc='upper right')
        
        # 保存图像
        output_filename = f"{raw_sample['scene_name']}_{raw_sample['image_name']}_{raw_sample['person_A_id']}_{raw_sample['person_B_id']}.png"
        output_filename = output_filename.replace('.jpg', '').replace('.png', '') + '.png'
        output_path = os.path.join(save_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 可视化图像已保存: {output_path}")
        
    except Exception as e:
        print(f"❌ 可视化过程出错: {e}")
        import traceback
        traceback.print_exc()


def plot_feature_distribution(samples, save_dir="./visualization_output"):
    """
    绘制特征分布图
    
    Args:
        samples: 样本列表
        save_dir: 保存目录
    """
    if not samples:
        return
    
    # 收集所有特征和标签
    all_features = []
    all_labels = []
    
    for sample in samples:
        features = sample['features'].numpy()
        label = sample['stage2_label'].item()
        all_features.append(features)
        all_labels.append(label)
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # 绘制特征分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 几何特征分布
    geometric_features = all_features[:, :7]
    axes[0, 0].hist(geometric_features[:, 0], bins=20, alpha=0.7, label='Distance')
    axes[0, 0].set_title('几何特征分布 - 距离')
    axes[0, 0].set_xlabel('Distance')
    axes[0, 0].set_ylabel('Frequency')
    
    # HoG特征分布
    if all_features.shape[1] > 7:
        hog_features = all_features[:, 7:71] if all_features.shape[1] >= 71 else all_features[:, 7:]
        hog_mean = np.mean(hog_features, axis=1)
        axes[0, 1].hist(hog_mean, bins=20, alpha=0.7, color='orange')
        axes[0, 1].set_title('HoG特征分布 - 平均值')
        axes[0, 1].set_xlabel('HoG Mean')
        axes[0, 1].set_ylabel('Frequency')
    
    # 标签分布
    label_names = ['Walking Together', 'Standing Together', 'Sitting Together']
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    axes[1, 0].bar([label_names[i] if i < len(label_names) else f'Class_{i}' for i in unique_labels], 
                  counts, alpha=0.7, color=['blue', 'green', 'red'][:len(unique_labels)])
    axes[1, 0].set_title('类别分布')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 特征维度统计
    feature_dims = [len(features) for features in all_features]
    axes[1, 1].hist(feature_dims, bins=10, alpha=0.7, color='purple')
    axes[1, 1].set_title('特征维度分布')
    axes[1, 1].set_xlabel('Feature Dimension')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(save_dir, 'feature_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 特征分布图已保存: {output_path}")


def main():
    """主测试函数"""
    print("🔍 开始数据载入过程可视化测试...")
    
    # 创建配置
    config = Stage2Config(
        data_path="../dataset",  # 根据实际路径调整
        batch_size=1,
        num_workers=0,  # 设为0便于调试
        frame_interval=10,  # 每10帧采样
        temporal_mode='none',  # Basic模式
        use_geometric=True,
        use_hog=True,
        use_scene_context=True
    )
    
    print(f"📋 配置信息:")
    print(f"  数据路径: {config.data_path}")
    print(f"  采样间隔: 每{config.frame_interval}帧")
    print(f"  特征配置: 几何={config.use_geometric}, HoG={config.use_hog}, 场景={config.use_scene_context}")
    print(f"  输入维度: {config.get_input_dim()}")
    
    try:
        # 创建数据集
        print(f"\n📂 创建训练数据集...")
        train_dataset = BasicStage2Dataset(
            data_path=config.data_path,
            split='train',
            use_geometric=config.use_geometric,
            use_hog=config.use_hog,
            use_scene_context=config.use_scene_context,
            frame_interval=config.frame_interval,
            use_oversampling=False  # 测试时不使用过采样
        )
        
        print(f"✅ 数据集加载成功!")
        print(f"  训练集大小: {len(train_dataset)}")
        
        # 可视化前10个样本
        print(f"\n🎨 开始可视化前10个交互对...")
        samples_to_visualize = []
        
        for i in range(min(10, len(train_dataset))):
            print(f"\n{'='*50}")
            print(f"处理第 {i+1} 个交互对:")
            
            try:
                sample = train_dataset[i]
                samples_to_visualize.append(sample)
                visualize_interaction_pair(sample, train_dataset, i)
                
            except Exception as e:
                print(f"❌ 处理第 {i+1} 个样本时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 绘制特征分布
        if samples_to_visualize:
            print(f"\n📈 绘制特征分布图...")
            plot_feature_distribution(samples_to_visualize)
        
        print(f"\n✅ 可视化测试完成!")
        print(f"  成功处理样本数: {len(samples_to_visualize)}")
        print(f"  可视化结果保存在: ./visualization_output/")
        
    except FileNotFoundError as e:
        print(f"❌ 数据路径错误: {e}")
        print("请检查config.data_path是否指向正确的数据集目录")
        
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()