#!/usr/bin/env python3
"""
Environment Check Script
Verifies that all dependencies and settings are correct for ResNet training
"""

import os
import sys

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

def check_python():
    """检查Python版本"""
    print("=== Python Environment ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    return True

def check_pytorch():
    """检查PyTorch"""
    print("\n=== PyTorch ===")
    try:
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
        print(f"TorchVision version: {torchvision.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch not found: {e}")
        return False

def check_dependencies():
    """检查其他依赖"""
    print("\n=== Other Dependencies ===")
    dependencies = [
        'numpy', 'PIL', 'json', 'argparse', 'collections'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"[OK] {dep}")
        except ImportError:
            print(f"[MISSING] {dep}")
            missing.append(dep)
    
    return len(missing) == 0

def check_project_structure():
    """检查项目结构"""
    print("\n=== Project Structure ===")
    
    required_files = [
        'configs/resnet_stage2_config.py',
        'models/resnet_feature_extractors.py',
        'models/resnet_stage2_classifier.py',
        'datasets/resnet_stage2_dataset.py',
        'utils/resnet_model_factory.py',
        'train_resnet_stage2.py',
        'run_resnet_training.py'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[MISSING] {file}")
            missing.append(file)
    
    return len(missing) == 0

def check_data_path():
    """检查数据路径"""
    print("\n=== Data Path ===")
    
    default_path = "../dataset"
    if os.path.exists(default_path):
        print(f"[OK] Default data path exists: {default_path}")
        
        # 检查一些场景
        scenes = ['bytes-cafe-2019-02-07_0', 'clark-center-2019-02-28_0']
        found_scenes = 0
        
        for scene in scenes:
            scene_path = os.path.join(default_path, scene)
            if os.path.exists(scene_path):
                print(f"  [OK] Found scene: {scene}")
                found_scenes += 1
            else:
                print(f"  [MISSING] Missing scene: {scene}")
        
        return found_scenes > 0
    else:
        print(f"[MISSING] Default data path not found: {default_path}")
        print("   You'll need to specify --data_path when training")
        return False

def test_import():
    """测试导入ResNet组件"""
    print("\n=== Import Test ===")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from configs.resnet_stage2_config import get_resnet18_config
        print("[OK] Config import successful")
        
        from models.resnet_feature_extractors import ResNetRelationFeatureFusion
        print("[OK] Feature extractor import successful")
        
        from models.resnet_stage2_classifier import ResNetRelationStage2Classifier
        print("[OK] Classifier import successful")
        
        # 测试配置创建
        config = get_resnet18_config()
        print("[OK] Config creation successful")
        
        return True
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主检查函数"""
    print("[CHECK] ResNet Training Environment Check")
    print("="*50)
    
    checks = [
        ("Python", check_python),
        ("PyTorch", check_pytorch), 
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Data Path", check_data_path),
        ("Import Test", test_import)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"[ERROR] {name} check failed with error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Environment Check Results: {passed}/{total} passed")
    
    if passed == total:
        print("[SUCCESS] All checks passed! Ready to start training.")
        print("\nTo start training:")
        print("  python run_resnet_training.py")
        print("  or")
        print("  train_resnet.bat")
    else:
        print("[ERROR] Some checks failed. Please fix the issues above.")
        print(f"Missing: {total - passed} components")
    
    print("="*50)

if __name__ == '__main__':
    main()