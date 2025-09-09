#!/usr/bin/env python3
"""
详细调试HoG特征值分布
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hog_features import HoGFeatureExtractor

def debug_hog_values():
    print("=== HoG特征值详细分析 ===")
    
    extractor = HoGFeatureExtractor(target_dim=64, use_simple_truncation=True)
    
    # 创建一个有内容的测试图像
    image = Image.new('RGB', (100, 100), (255, 255, 255))  # 白色背景
    # 添加一些黑色方块制造特征
    import numpy as np
    img_array = np.array(image)
    img_array[20:30, 20:30] = [0, 0, 0]  # 黑色方块1
    img_array[60:80, 40:60] = [0, 0, 0]  # 黑色方块2
    img_array[40:45, 70:90] = [128, 128, 128]  # 灰色条纹
    image = Image.fromarray(img_array)
    
    person_A_box = torch.tensor([15, 15, 20, 20], dtype=torch.float32)
    person_B_box = torch.tensor([55, 35, 30, 50], dtype=torch.float32)
    
    print("步骤1: 计算联合边界框")
    combined_box = extractor.compute_smart_combined_box(person_A_box, person_B_box)
    print(f"联合边界框: {combined_box}")
    
    print("\n步骤2: 裁剪联合区域")  
    joint_region = image.crop(combined_box)
    print(f"联合区域尺寸: {joint_region.size}")
    
    print("\n步骤3: Resize区域")
    resized_region = extractor.resize_with_aspect_ratio(joint_region)
    print(f"Resize后尺寸: {resized_region.size}")
    
    print("\n步骤4: 提取原始HoG")
    raw_hog = extractor.extract_raw_hog(resized_region)
    if raw_hog is not None:
        print(f"原始HoG维度: {len(raw_hog)}")
        print(f"原始HoG统计: min={raw_hog.min():.6f}, max={raw_hog.max():.6f}, mean={raw_hog.mean():.6f}")
        print(f"原始HoG前10位: {raw_hog[:10]}")
        print(f"原始HoG零值比例: {(raw_hog == 0).sum() / len(raw_hog) * 100:.2f}%")
        
        print("\n步骤5: 简单截断到64维")
        if len(raw_hog) >= 64:
            truncated = raw_hog[:64]
            print(f"截断后统计: min={truncated.min():.6f}, max={truncated.max():.6f}, mean={truncated.mean():.6f}")
            print(f"截断后前10位: {truncated[:10]}")
            print(f"截断后零值比例: {(truncated == 0).sum() / len(truncated) * 100:.2f}%")
        
        print("\n步骤6: 完整流程测试")
        final_hog = extractor.extract_joint_hog_features(image, person_A_box, person_B_box)
        print(f"最终HoG统计: min={final_hog.min():.6f}, max={final_hog.max():.6f}, mean={final_hog.mean():.6f}")
        print(f"最终HoG零值比例: {(final_hog == 0).sum() / len(final_hog) * 100:.2f}%")
        
        if torch.all(final_hog == 0):
            print("[ERROR] 最终HoG全为零!")
        
    else:
        print("[ERROR] 原始HoG提取失败!")

def test_different_images():
    """测试不同类型的图像"""
    print("\n=== 测试不同图像类型 ===")
    
    extractor = HoGFeatureExtractor(target_dim=64, use_simple_truncation=True)
    person_A_box = torch.tensor([10, 10, 30, 40], dtype=torch.float32)
    person_B_box = torch.tensor([50, 20, 25, 35], dtype=torch.float32)
    
    test_images = [
        ("纯色图像", Image.new('RGB', (100, 100), (128, 128, 128))),
        ("随机噪声", None),  # 稍后生成
        ("梯度图像", None),  # 稍后生成
    ]
    
    # 生成随机噪声图像
    noise_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    test_images[1] = ("随机噪声", Image.fromarray(noise_array))
    
    # 生成梯度图像
    gradient_array = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        gradient_array[:, i] = [int(i * 2.55), int(i * 2.55), int(i * 2.55)]
    test_images[2] = ("梯度图像", Image.fromarray(gradient_array))
    
    for name, image in test_images:
        print(f"\n测试 {name}:")
        final_hog = extractor.extract_joint_hog_features(image, person_A_box, person_B_box)
        zero_ratio = (final_hog == 0).sum().item() / len(final_hog) * 100
        print(f"  零值比例: {zero_ratio:.1f}%")
        print(f"  特征范围: [{final_hog.min():.4f}, {final_hog.max():.4f}]")

if __name__ == '__main__':
    try:
        debug_hog_values()
        test_different_images()
        print("\n[SUCCESS] HoG值调试完成!")
        
    except Exception as e:
        print(f"[ERROR] 调试失败: {e}")
        import traceback
        traceback.print_exc()