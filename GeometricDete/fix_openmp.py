#!/usr/bin/env python3
"""
OpenMP Environment Fix
Fixes the libiomp5md.dll initialization error
"""

import os
import sys

def fix_openmp_error():
    """修复OpenMP重复初始化错误"""
    print("Fixing OpenMP duplicate library error...")
    
    # 方法1: 设置环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print("✅ Set KMP_DUPLICATE_LIB_OK=TRUE")
    
    # 方法2: 设置OMP线程数
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '4'
        print("✅ Set OMP_NUM_THREADS=4")
    
    # 方法3: 禁用MKL多线程（如果使用Intel MKL）
    os.environ['MKL_NUM_THREADS'] = '1'
    print("✅ Set MKL_NUM_THREADS=1")
    
    print("OpenMP environment fixed!")

def check_environment():
    """检查当前环境设置"""
    print("Current environment settings:")
    
    env_vars = [
        'KMP_DUPLICATE_LIB_OK',
        'OMP_NUM_THREADS', 
        'MKL_NUM_THREADS',
        'NUMBA_NUM_THREADS'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    print("\nPython path:")
    print(f"  {sys.executable}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("\nPyTorch not found")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        check_environment()
    else:
        fix_openmp_error()
        check_environment()