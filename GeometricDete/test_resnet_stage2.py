#!/usr/bin/env python3
"""
Universal ResNet Stage2 Model Testing Script
Tests any ResNet backbone (resnet18/34/50) on test dataset
Provides Accuracy, MPCA, and FLOPs metrics
"""

# 修复OpenMP错误（必须在导入torch之前）
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from datetime import datetime
import json
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入组件
from configs.resnet_stage2_config import get_resnet18_config, get_resnet50_config
from datasets.resnet_stage2_dataset import create_resnet_stage2_data_loaders
from utils.resnet_model_factory import create_resnet_stage2_model, ResNetModelCheckpointManager
from geometric_stage2_classifier import Stage2Evaluator

# 用于FLOPs计算
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop not installed. FLOPs calculation will be skipped.")
    print("Install with: pip install thop")
    THOP_AVAILABLE = False


class ResNetStage2Tester:
    """Universal ResNet Stage2 model tester"""
    
    def __init__(self, model_path: str, device: torch.device, data_path: str = "../dataset"):
        self.model_path = model_path
        self.device = device
        self.data_path = data_path
        
        # 加载模型和配置
        self.model, self.config = self._load_model_and_config()
        self.model = self.model.to(device)
        self.model.eval()
        
        # 创建数据加载器
        self._create_data_loaders()
        
        print(f"ResNet Stage2 Tester initialized:")
        print(f"  Model: {self.config.backbone_name}")
        print(f"  Device: {device}")
        print(f"  Test samples: {len(self.test_loader.dataset)}")
    
    def _load_model_and_config(self):
        """加载模型和配置"""
        print(f"Loading model from: {self.model_path}")
        
        # 加载checkpoint
        checkpoint_manager = ResNetModelCheckpointManager(os.path.dirname(self.model_path))
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # 从checkpoint中获取配置信息
        if 'config' in checkpoint:
            backbone_name = checkpoint['config'].get('backbone_name', 'resnet18')
            visual_feature_dim = checkpoint['config'].get('visual_feature_dim', 256)
        else:
            # 如果没有配置信息，尝试从模型信息推断
            model_info = checkpoint.get('model_info', {})
            backbone_name = 'resnet18'  # 默认值
            visual_feature_dim = 256
            print("Warning: No config found in checkpoint, using defaults")
        
        # 根据backbone类型创建对应配置
        if backbone_name == 'resnet50':
            config = get_resnet50_config()
        else:
            config = get_resnet18_config()
            if backbone_name == 'resnet34':
                config.backbone_name = 'resnet34'
        
        # 更新配置
        config.visual_feature_dim = visual_feature_dim
        config.data_path = self.data_path
        
        # 创建模型
        model = create_resnet_stage2_model(config)
        
        # 加载权重
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded model: {backbone_name}")
        print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   Metrics: {checkpoint.get('metrics', {})}")
        
        return model, config
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_resnet_stage2_data_loaders(self.config)
        self.test_loader = test_loader
        print(f"✅ Test loader created: {len(test_loader)} batches")
    
    def calculate_flops(self):
        """计算模型FLOPs"""
        if not THOP_AVAILABLE:
            return None, None, None
        
        print("\n🔧 Calculating FLOPs...")
        
        # 创建示例输入
        batch_size = 1
        person_feature_dim = self.config.get_person_feature_dim()
        spatial_feature_dim = self.config.get_spatial_feature_dim()
        
        # 检查模型是否期望图像输入还是特征输入
        # 我们测试特征输入（预计算特征的情况）
        person_A_features = torch.randn(batch_size, person_feature_dim).to(self.device)
        person_B_features = torch.randn(batch_size, person_feature_dim).to(self.device)
        spatial_features = torch.randn(batch_size, spatial_feature_dim).to(self.device)
        
        try:
            # 计算FLOPs
            model_for_profile = self.model.module if hasattr(self.model, 'module') else self.model
            flops, params = profile(
                model_for_profile, 
                inputs=(person_A_features, person_B_features, spatial_features),
                verbose=False
            )
            
            # 格式化输出
            flops_str, params_str = clever_format([flops, params], "%.3f")
            
            print(f"✅ FLOPs calculation completed:")
            print(f"   FLOPs: {flops_str}")
            print(f"   Params: {params_str}")
            
            return flops, params, flops_str, params_str
            
        except Exception as e:
            print(f"❌ FLOPs calculation failed: {e}")
            return None, None, None, None
    
    def test_model(self):
        """在测试集上评估模型"""
        print(f"\n🧪 Testing model on test dataset...")
        
        self.model.eval()
        test_loss = 0
        total_samples = 0
        
        # 创建评估器
        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        evaluator = Stage2Evaluator(class_names)
        
        # 按场景统计分类情况
        scene_stats = {}  # {scene_name: {true_label: {pred_label: count}}}
        scene_samples = {}  # {scene_name: total_count}
        
        # 创建损失函数（用于计算损失）
        from utils.resnet_model_factory import create_resnet_stage2_loss
        criterion = create_resnet_stage2_loss(self.config).to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # 数据移动到设备
                def normalize_person(t):
                    if torch.is_tensor(t) and t.dim() == 5:
                        B, T, C, H, W = t.shape
                        return t.squeeze(1) if T == 1 else t.mean(dim=1)
                    return t

                person_A_features = normalize_person(batch['person_A_features']).to(self.device)
                person_B_features = normalize_person(batch['person_B_features']).to(self.device)
                spatial_features = batch['spatial_features'].to(self.device)
                targets = batch['stage2_label'].to(self.device)
                
                # 获取场景信息
                scene_names = batch.get('scene_name', ['unknown'] * targets.size(0))
                if isinstance(scene_names, str):
                    scene_names = [scene_names]
                
                # 前向传播
                logits = self.model(person_A_features, person_B_features, spatial_features)
                loss, _ = criterion(logits, targets)
                
                # 累计损失
                test_loss += loss.item()
                total_samples += targets.size(0)
                
                # 获取预测结果
                predictions = torch.argmax(logits, dim=1)
                
                # 更新评估器
                evaluator.update(predictions.cpu().numpy(), targets.cpu().numpy())
                
                # 按场景统计分类情况
                pred_np = predictions.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                for i in range(len(targets_np)):
                    scene_name = scene_names[i] if i < len(scene_names) else 'unknown'
                    true_label = int(targets_np[i])
                    pred_label = int(pred_np[i])
                    
                    # 初始化场景统计
                    if scene_name not in scene_stats:
                        scene_stats[scene_name] = {}
                        scene_samples[scene_name] = 0
                    
                    if true_label not in scene_stats[scene_name]:
                        scene_stats[scene_name][true_label] = {}
                    
                    if pred_label not in scene_stats[scene_name][true_label]:
                        scene_stats[scene_name][true_label][pred_label] = 0
                    
                    scene_stats[scene_name][true_label][pred_label] += 1
                    scene_samples[scene_name] += 1
                
                # 打印进度
                if batch_idx % 10 == 0:
                    print(f'  Progress: [{batch_idx:3d}/{len(self.test_loader)}] '
                          f'({100. * batch_idx / len(self.test_loader):3.0f}%)')
        
        # 计算指标
        test_time = time.time() - start_time
        avg_loss = test_loss / len(self.test_loader)
        test_metrics = evaluator.compute_metrics()
        
        # 提取关键指标
        accuracy = test_metrics.get('overall_accuracy', 0.0)
        mpca = test_metrics.get('mpca', 0.0)
        per_class_acc = test_metrics.get('per_class_accuracy', [])
        
        # 处理场景统计数据
        scene_analysis = self._analyze_scene_stats(scene_stats, scene_samples, class_names)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'mpca': mpca,
            'per_class_accuracy': per_class_acc,
            'detailed_metrics': test_metrics,
            'test_time': test_time,
            'total_samples': total_samples,
            'scene_stats': scene_stats,
            'scene_analysis': scene_analysis
        }
    
    def _analyze_scene_stats(self, scene_stats, scene_samples, class_names):
        """分析场景统计数据"""
        scene_analysis = {}
        
        for scene_name, stats in scene_stats.items():
            scene_analysis[scene_name] = {
                'total_samples': scene_samples[scene_name],
                'per_class': {},
                'confusion_matrix': [[0 for _ in range(3)] for _ in range(3)],
                'accuracy': 0.0
            }
            
            correct_predictions = 0
            total_scene_samples = scene_samples[scene_name]
            
            # 计算每个真实标签的统计
            for true_label in range(3):
                class_name = class_names[true_label]
                scene_analysis[scene_name]['per_class'][class_name] = {
                    'true_samples': 0,
                    'correct': 0,
                    'wrong': 0,
                    'accuracy': 0.0,
                    'predictions': {class_names[i]: 0 for i in range(3)}
                }
                
                if true_label in stats:
                    true_samples = sum(stats[true_label].values())
                    scene_analysis[scene_name]['per_class'][class_name]['true_samples'] = true_samples
                    
                    for pred_label, count in stats[true_label].items():
                        pred_class_name = class_names[pred_label]
                        scene_analysis[scene_name]['per_class'][class_name]['predictions'][pred_class_name] = count
                        scene_analysis[scene_name]['confusion_matrix'][true_label][pred_label] = count
                        
                        if pred_label == true_label:
                            scene_analysis[scene_name]['per_class'][class_name]['correct'] = count
                            correct_predictions += count
                        else:
                            scene_analysis[scene_name]['per_class'][class_name]['wrong'] += count
                    
                    # 计算该类别在该场景的准确率
                    if true_samples > 0:
                        scene_analysis[scene_name]['per_class'][class_name]['accuracy'] = \
                            scene_analysis[scene_name]['per_class'][class_name]['correct'] / true_samples
            
            # 计算场景总体准确率
            if total_scene_samples > 0:
                scene_analysis[scene_name]['accuracy'] = correct_predictions / total_scene_samples
        
        return scene_analysis
    
    def _print_scene_analysis(self, scene_analysis):
        """打印场景分析结果"""
        print(f"\nScene-wise Analysis:")
        print(f"{'─'*80}")
        
        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        
        # 按场景排序
        sorted_scenes = sorted(scene_analysis.items(), key=lambda x: x[1]['total_samples'], reverse=True)
        
        for scene_name, analysis in sorted_scenes:
            print(f"\n📍 Scene: {scene_name}")
            print(f"   Total samples: {analysis['total_samples']}")
            print(f"   Scene accuracy: {analysis['accuracy']:.4f} ({analysis['accuracy']*100:.2f}%)")
            
            # 混淆矩阵
            print(f"   Confusion Matrix:")
            print(f"        {'':>12} {'Predicted':>36}")
            print(f"   {'True':>8} {'W.T.':>8} {'S.T.':>8} {'Sit.T.':>8} {'Total':>8}")
            for i in range(3):
                true_class = class_names[i][:4]  # 缩短类名
                row = analysis['confusion_matrix'][i]
                total = sum(row)
                print(f"   {true_class:>8} {row[0]:>8} {row[1]:>8} {row[2]:>8} {total:>8}")
            
            # 每类详细统计
            print(f"   Per-class performance:")
            for class_name, stats in analysis['per_class'].items():
                if stats['true_samples'] > 0:
                    short_name = class_name.replace(' Together', '').replace('Walking', 'Walk').replace('Standing', 'Stand').replace('Sitting', 'Sit')
                    correct = stats['correct']
                    wrong = stats['wrong']
                    total = stats['true_samples']
                    acc = stats['accuracy']
                    
                    print(f"     {short_name:>12}: {correct:>3}/{total:>3} correct ({acc:.3f}) | Errors: {wrong:>3}")
                    
                    # 显示错误分类情况
                    errors = []
                    for pred_class, count in stats['predictions'].items():
                        if pred_class != class_name and count > 0:
                            pred_short = pred_class.replace(' Together', '').replace('Walking', 'Walk').replace('Standing', 'Stand').replace('Sitting', 'Sit')
                            errors.append(f"{count}→{pred_short}")
                    
                    if errors:
                        print(f"     {'':<12}   Misclassified as: {', '.join(errors)}")
        
        # 汇总所有场景的表现
        print(f"\n📊 Scene Performance Summary:")
        scene_performances = [(name, analysis['accuracy']) for name, analysis in scene_analysis.items()]
        scene_performances.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Best performing scenes:")
        for i, (scene_name, acc) in enumerate(scene_performances[:3]):
            print(f"     {i+1}. {scene_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        if len(scene_performances) > 3:
            print(f"   Worst performing scenes:")
            for i, (scene_name, acc) in enumerate(scene_performances[-3:]):
                print(f"     {len(scene_performances)-2+i}. {scene_name}: {acc:.4f} ({acc*100:.2f}%)")
    
    def generate_report(self, test_results, flops_results=None):
        """生成测试报告"""
        print(f"\n{'='*80}")
        print(f"RESNET STAGE2 TEST REPORT")
        print(f"{'='*80}")
        
        # 模型信息
        print(f"Model Information:")
        print(f"  Backbone: {self.config.backbone_name}")
        print(f"  Visual Features: {self.config.visual_feature_dim}D")
        print(f"  Fusion Strategy: {self.config.fusion_strategy}")
        print(f"  Model Path: {self.model_path}")
        
        # 硬件需求（FLOPs）
        print(f"\nHardware Requirements:")
        if flops_results and flops_results[0] is not None:
            flops, params, flops_str, params_str = flops_results
            print(f"  FLOPs: {flops_str} ({flops:,.0f})")
            print(f"  Parameters: {params_str} ({params:,.0f})")
            print(f"  Memory (params): ~{params * 4 / 1024 / 1024:.1f} MB")
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Parameters: {total_params:,}")
            print(f"  Memory (params): ~{total_params * 4 / 1024 / 1024:.1f} MB")
            print(f"  FLOPs: Not available (install thop)")
        
        # 测试结果
        print(f"\nTest Results:")
        print(f"  Dataset: {len(self.test_loader.dataset)} samples")
        print(f"  Test Time: {test_results['test_time']:.2f}s")
        print(f"  Throughput: {test_results['total_samples'] / test_results['test_time']:.1f} samples/sec")
        print(f"  Average Loss: {test_results['loss']:.6f}")
        
        # 主要指标
        print(f"\nKey Metrics:")
        print(f"  Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
        print(f"  MPCA: {test_results['mpca']:.4f} ({test_results['mpca']*100:.2f}%)")
        
        # 每类准确率
        print(f"\nPer-Class Accuracy:")
        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        for i, (cls_name, acc) in enumerate(zip(class_names, test_results['per_class_accuracy'])):
            print(f"  {cls_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        # 场景分析
        if 'scene_analysis' in test_results:
            self._print_scene_analysis(test_results['scene_analysis'])
        
        print(f"{'='*80}")
        
        return {
            'model_info': {
                'backbone': self.config.backbone_name,
                'visual_feature_dim': self.config.visual_feature_dim,
                'fusion_strategy': self.config.fusion_strategy,
                'model_path': str(self.model_path)
            },
            'hardware_requirements': {
                'flops': flops_results[0] if flops_results and flops_results[0] else None,
                'flops_str': flops_results[2] if flops_results and flops_results[2] else None,
                'parameters': flops_results[1] if flops_results and flops_results[1] else sum(p.numel() for p in self.model.parameters()),
                'params_str': flops_results[3] if flops_results and flops_results[3] else None
            },
            'test_results': test_results,
            'scene_analysis': test_results.get('scene_analysis', {}),
            'timestamp': datetime.now().isoformat()
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Universal ResNet Stage2 Model Testing')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_path', type=str, default='../dataset',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for test results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--save_report', action='store_true',
                       help='Save detailed report to JSON file')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查模型文件
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        return
    
    try:
        # 创建测试器
        tester = ResNetStage2Tester(
            model_path=str(model_path),
            device=device,
            data_path=args.data_path
        )
        
        # 计算FLOPs
        flops_results = tester.calculate_flops()
        
        # 测试模型
        test_results = tester.test_model()
        
        # 生成报告
        report = tester.generate_report(test_results, flops_results)
        
        # 保存报告
        if args.save_report:
            os.makedirs(args.output_dir, exist_ok=True)
            model_name = model_path.stem
            report_path = Path(args.output_dir) / f"{model_name}_test_report.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n💾 Test report saved to: {report_path}")
        
        print(f"\n✅ Testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()