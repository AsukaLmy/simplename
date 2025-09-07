#!/usr/bin/env python3
"""
计算HoG特征的确切维度和模型参数量
"""

def calculate_hog_dimensions():
    """计算HoG特征维度"""
    
    # HoG配置（来自hog_features.py）
    orientations = 8
    pixels_per_cell = (12, 12)
    cells_per_block = (2, 2)
    
    print("=== HoG维度计算 ===")
    print(f"orientations: {orientations}")
    print(f"pixels_per_cell: {pixels_per_cell}")
    print(f"cells_per_block: {cells_per_block}")
    
    # 假设图像resize后的尺寸（需要查看实际代码）
    # 从hog_features.py中的resize_with_aspect_ratio推测
    typical_sizes = [
        (64, 64),    # 小尺寸
        (96, 96),    # 中等尺寸  
        (128, 128),  # 大尺寸
        (160, 160),  # 更大尺寸
    ]
    
    print(f"\n不同图像尺寸下的HoG维度:")
    for width, height in typical_sizes:
        # 计算cell数量
        cells_x = width // pixels_per_cell[0]
        cells_y = height // pixels_per_cell[1] 
        
        # 计算block数量（滑动窗口）
        blocks_x = cells_x - cells_per_block[0] + 1
        blocks_y = cells_y - cells_per_block[1] + 1
        
        # 确保blocks数量非负
        blocks_x = max(0, blocks_x)
        blocks_y = max(0, blocks_y)
        
        # 总维度 = blocks * cells_per_block * orientations
        total_dim = blocks_x * blocks_y * (cells_per_block[0] * cells_per_block[1]) * orientations
        
        print(f"  {width}x{height}: cells=({cells_x},{cells_y}), blocks=({blocks_x},{blocks_y}), HoG维度={total_dim}")
    
    return total_dim

def calculate_model_parameters(input_dim, hidden_dims=[128, 64, 32], num_classes=3):
    """计算模型参数量"""
    
    print(f"\n=== 模型参数量计算 ===")
    print(f"输入维度: {input_dim}")
    print(f"隐藏层: {hidden_dims}")
    print(f"输出类别: {num_classes}")
    
    total_params = 0
    
    # 输入层: input_dim -> hidden_dims[0]
    input_layer_params = input_dim * hidden_dims[0] + hidden_dims[0]  # weights + bias
    total_params += input_layer_params
    print(f"\n输入层参数: {input_dim} × {hidden_dims[0]} + {hidden_dims[0]} = {input_layer_params:,}")
    
    # 隐藏层
    prev_dim = hidden_dims[0]
    for i, hidden_dim in enumerate(hidden_dims[1:], 1):
        layer_params = prev_dim * hidden_dim + hidden_dim
        total_params += layer_params
        print(f"隐藏层{i}参数: {prev_dim} × {hidden_dim} + {hidden_dim} = {layer_params:,}")
        prev_dim = hidden_dim
    
    # 输出层: last_hidden -> num_classes
    output_layer_params = prev_dim * num_classes + num_classes
    total_params += output_layer_params
    print(f"输出层参数: {prev_dim} × {num_classes} + {num_classes} = {output_layer_params:,}")
    
    print(f"\n总参数量: {total_params:,}")
    return total_params

def compare_scenarios():
    """比较不同特征配置下的参数量"""
    
    print(f"\n=== 参数量对比 ===")
    
    # 当前配置（64维HoG）
    current_input_dim = 7 + 64 + 1  # 几何 + HoG + 场景
    current_params = calculate_model_parameters(current_input_dim)
    
    # 原始HoG配置（假设2592维）
    full_hog_dim = 2592  # 从128x128图像计算得出
    full_input_dim = 7 + full_hog_dim + 1  # 几何 + 完整HoG + 场景
    full_params = calculate_model_parameters(full_input_dim)
    
    # 无HoG配置
    no_hog_input_dim = 7 + 1  # 仅几何 + 场景
    no_hog_params = calculate_model_parameters(no_hog_input_dim)
    
    print(f"\n=== 对比总结 ===")
    print(f"当前配置 (7+64+1=72维):     {current_params:,} 参数")
    print(f"完整HoG (7+{full_hog_dim}+1={full_input_dim}维): {full_params:,} 参数 (增加 {full_params-current_params:,})")
    print(f"无HoG配置 (7+1=8维):        {no_hog_params:,} 参数 (减少 {current_params-no_hog_params:,})")
    
    print(f"\n参数量比例:")
    print(f"完整HoG / 当前: {full_params/current_params:.1f}x")
    print(f"当前 / 无HoG: {current_params/no_hog_params:.1f}x")
    
    return {
        'current': current_params,
        'full_hog': full_params,
        'no_hog': no_hog_params
    }

if __name__ == '__main__':
    # 计算HoG维度
    hog_dim = calculate_hog_dimensions()
    
    # 比较参数量
    params = compare_scenarios()
    
    print(f"\n=== 推荐方案 ===")
    if params['full_hog'] > 1000000:  # 超过100万参数
        print("⚠️  完整HoG会导致参数量过大，可能导致:")
        print("   - 训练速度显著下降")
        print("   - 内存占用增加")
        print("   - 过拟合风险增加")
        print("   - 推理速度下降")
        print(f"\n💡 建议方案:")
        print("   1. 保持当前64维HoG (最平衡)")
        print("   2. 简单截断到64维 (去除PCA开销)")
        print("   3. 完全移除HoG (最高性能)")
    else:
        print("✅ 完整HoG参数量可接受")