#!/usr/bin/env python3
"""
ResNet-based Stage2 Behavior Classification Training Script
Complete training pipeline for ResNet backbone with Relation Network
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入ResNet相关组件
from configs.resnet_stage2_config import get_resnet18_config, get_resnet50_config
from datasets.resnet_stage2_dataset import create_resnet_stage2_data_loaders
from utils.resnet_model_factory import create_resnet_training_setup, ResNetModelCheckpointManager
from models.resnet_stage2_classifier import ResNetRelationStage2Classifier
from geometric_stage2_classifier import Stage2Evaluator  # 复用评估器

# 导入FLOP计算库
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop not available. FLOPs calculation will be skipped.")
    THOP_AVAILABLE = False



class ResNetStage2Trainer:
    """ResNet-based Stage2 behavior classification trainer"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # 创建检查点管理器
        self.checkpoint_manager = ResNetModelCheckpointManager(config.checkpoint_dir)
        
        # 创建模型和损失，但延迟创建optimizer以便先应用冻结策略
        from utils.resnet_model_factory import create_resnet_stage2_model, create_resnet_stage2_loss, create_resnet_optimizer, create_resnet_scheduler

        # 创建模型并移动到设备
        self.model = create_resnet_stage2_model(config).to(device)

        # 应用冻结策略（冻结前 N 个残差块并把 BN 设为 eval）
        model_for_modify = self.model.module if hasattr(self.model, 'module') else self.model
        freeze_blocks = getattr(config, 'freeze_blocks', 0)
        if freeze_blocks and hasattr(model_for_modify.backbone, 'freeze_early_layers'):
            print(f"Applying freeze: first {freeze_blocks} residual blocks")
            model_for_modify.backbone.freeze_early_layers(freeze_layers=freeze_blocks)
            import torch.nn as _nn
            for m in model_for_modify.backbone.modules():
                if isinstance(m, _nn.BatchNorm2d):
                    m.eval()

            # Debug: print backbone parameter counts and a few trainable param names
            try:
                bp_total = sum(p.numel() for p in model_for_modify.backbone.parameters())
                bp_train = sum(p.numel() for p in model_for_modify.backbone.parameters() if p.requires_grad)
                print(f"[DEBUG] Backbone params: {bp_total:,}, trainable after freeze: {bp_train:,}")
                ct = 0
                for n, p in model_for_modify.named_parameters():
                    if n.startswith('backbone') and p.requires_grad:
                        print(f"  trainable param: {n}  shape={tuple(p.shape)}")
                        ct += 1
                        if ct >= 20:
                            break
            except Exception:
                pass
            # Ensure freeze actually applied: fallback to name-based freezing
            try:
                layer_names = []
                # map freeze_blocks to layer name fragments
                for i in range(1, freeze_blocks + 1):
                    layer_names.append(f'backbone.backbone.layer{i}')
                # also freeze initial conv/bn when freezing >=1
                if freeze_blocks >= 1:
                    layer_names.extend(['backbone.backbone.conv1', 'backbone.backbone.bn1'])

                frozen_any = False
                for n, p in model_for_modify.named_parameters():
                    if any(ln in n for ln in layer_names):
                        if p.requires_grad:
                            p.requires_grad = False
                            frozen_any = True
                if frozen_any:
                    bp_total = sum(p.numel() for p in model_for_modify.backbone.parameters())
                    bp_train = sum(p.numel() for p in model_for_modify.backbone.parameters() if p.requires_grad)
                    print(f"[DEBUG-Fallback] Backbone params: {bp_total:,}, trainable after fallback freeze: {bp_train:,}")
            except Exception:
                pass

        # 创建损失
        self.criterion = create_resnet_stage2_loss(config).to(device)

        # 现在创建优化器和调度器（会检测 model.backbone 并分组）
        self.optimizer = create_resnet_optimizer(self.model, config)
        self.scheduler = create_resnet_scheduler(self.optimizer, config)
        
        # 训练记录
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_mpcas = []
        
        # 详细的epoch记录
        self.epoch_results = []
        
        # 最佳模型记录
        self.best_val_mpca = 0.0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        print(f"ResNet Stage2 Trainer initialized on {device}")
        print(f"Model: {config.backbone_name}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_mpca_loss = 0
        epoch_acc = 0
        
        # 创建评估器
        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        evaluator = Stage2Evaluator(class_names)
        
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # 数据移动到设备
            # Normalize possible temporal dim: [B, T, 3, H, W] -> [B, 3, H, W]
            def normalize_person(t):
                if torch.is_tensor(t):
                    if t.dim() == 5:
                        B, T, C, H, W = t.shape
                        if T == 1:
                            t = t.squeeze(1)
                        else:
                            t = t.mean(dim=1)
                return t

            person_A_features = normalize_person(batch['person_A_features']).to(self.device)
            person_B_features = normalize_person(batch['person_B_features']).to(self.device)
            spatial_features = batch['spatial_features'].to(self.device)
            targets = batch['stage2_label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            try:
                logits = self.model(person_A_features, person_B_features, spatial_features)
            except Exception as e:
                print(f"Forward failed on batch {batch_idx}; shapes:")
                try:
                    print(' person_A:', None if person_A_features is None else tuple(person_A_features.shape))
                    print(' person_B:', None if person_B_features is None else tuple(person_B_features.shape))
                    print(' spatial:', None if spatial_features is None else tuple(spatial_features.shape))
                except Exception:
                    pass
                raise
            loss, loss_dict = self.criterion(logits, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录损失
            epoch_loss += loss.item()
            epoch_ce_loss += loss_dict['ce_loss']
            epoch_mpca_loss += loss_dict['mpca_loss']
            epoch_acc += loss_dict['overall_acc']
            
            # 更新评估器
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                evaluator.update(predictions.cpu().numpy(), targets.cpu().numpy())
            
            # 打印进度
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(targets):5d}/'
                      f'{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):3.0f}%)] '
                      f'Loss: {loss.item():.6f}')
        
        # 计算平均值
        avg_loss = epoch_loss / len(train_loader)
        avg_ce_loss = epoch_ce_loss / len(train_loader)
        avg_mpca_loss = epoch_mpca_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        
        # 计算详细评估指标
        train_metrics = evaluator.compute_metrics()
        train_mpca = train_metrics.get('mpca', 0.0)
        
        epoch_time = time.time() - start_time
        
        print(f'Train Epoch {epoch}: Avg Loss: {avg_loss:.6f}, '
              f'Acc: {avg_acc:.4f}, MPCA: {train_mpca:.4f}, Time: {epoch_time:.1f}s')
        
        return avg_loss, avg_acc, train_mpca, {
            'ce_loss': avg_ce_loss,
            'mpca_loss': avg_mpca_loss,
            'detailed_metrics': train_metrics
        }
    
    def validate_epoch(self, val_loader, epoch):
        """验证一个epoch"""
        self.model.eval()
        val_loss = 0
        
        # 创建评估器
        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        evaluator = Stage2Evaluator(class_names)
        
        with torch.no_grad():
            for batch in val_loader:
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
                
                # 前向传播
                logits = self.model(person_A_features, person_B_features, spatial_features)
                loss, _ = self.criterion(logits, targets)
                val_loss += loss.item()
                
                # 更新评估器
                predictions = torch.argmax(logits, dim=1)
                evaluator.update(predictions.cpu().numpy(), targets.cpu().numpy())
        
        # 计算指标
        avg_loss = val_loss / len(val_loader)
        val_metrics = evaluator.compute_metrics()
        val_acc = val_metrics.get('overall_accuracy', 0.0)
        val_mpca = val_metrics.get('mpca', 0.0)
        
        print(f'Val Epoch {epoch}: Avg Loss: {avg_loss:.6f}, '
              f'Acc: {val_acc:.4f}, MPCA: {val_mpca:.4f}')
        
        return avg_loss, val_acc, val_mpca, val_metrics
    
    def test_epoch(self, test_loader):
        """测试模型"""
        self.model.eval()
        test_loss = 0
        
        # 创建评估器
        class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
        evaluator = Stage2Evaluator(class_names)
        
        print(f"\nEvaluating on test set...")
        with torch.no_grad():
            for batch in test_loader:
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
                
                # 前向传播
                logits = self.model(person_A_features, person_B_features, spatial_features)
                loss, _ = self.criterion(logits, targets)
                test_loss += loss.item()
                
                # 更新评估器
                predictions = torch.argmax(logits, dim=1)
                evaluator.update(predictions.cpu().numpy(), targets.cpu().numpy())
        
        # 计算指标
        avg_loss = test_loss / len(test_loader)
        test_metrics = evaluator.compute_metrics()
        test_acc = test_metrics.get('overall_accuracy', 0.0)
        test_mpca = test_metrics.get('mpca', 0.0)
        
        # 打印详细结果
        evaluator.print_evaluation_report()
        
        return avg_loss, test_acc, test_mpca, test_metrics
    
    def save_and_print_epoch_results(self, epoch, train_loss, train_acc, train_mpca, train_details, 
                                   val_loss=None, val_acc=None, val_mpca=None, val_metrics=None):
        """保存和打印epoch结果"""
        epoch_result = {
            'epoch': epoch,
            'train': {
                'loss': train_loss,
                'accuracy': train_acc,
                'mpca': train_mpca,
                'ce_loss': train_details.get('ce_loss', 0.0),
                'mpca_loss': train_details.get('mpca_loss', 0.0),
                'detailed_metrics': train_details.get('detailed_metrics', {})
            }
        }
        
        if val_loss is not None:
            epoch_result['val'] = {
                'loss': val_loss,
                'accuracy': val_acc,
                'mpca': val_mpca,
                'detailed_metrics': val_metrics or {}
            }
        
        # 添加到记录中
        self.epoch_results.append(epoch_result)
        
        # 打印格式化结果
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} RESULTS")
        print(f"{'='*80}")
        print(f"Train - Loss: {train_loss:.6f} | Acc: {train_acc:.4f} | MPCA: {train_mpca:.4f}")
        print(f"        CE Loss: {train_details.get('ce_loss', 0.0):.6f} | MPCA Loss: {train_details.get('mpca_loss', 0.0):.6f}")
        
        if val_loss is not None:
            print(f"Val   - Loss: {val_loss:.6f} | Acc: {val_acc:.4f} | MPCA: {val_mpca:.4f}")
            
            # 打印详细的验证指标（如果有的话）
            if val_metrics and 'per_class_accuracy' in val_metrics:
                print("        Per-class accuracy:")
                class_names = ['Walking Together', 'Standing Together', 'Sitting Together']
                for i, (cls_name, acc) in enumerate(zip(class_names, val_metrics['per_class_accuracy'])):
                    print(f"          {cls_name}: {acc:.4f}")
        
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*80}")
        
        # 每10个epoch或最后一个epoch时，保存到文件
        if epoch % 10 == 0 or epoch == self.config.epochs:
            self.save_results_to_file()
    
    def calculate_flops(self):
        """计算模型FLOPs"""
        if not THOP_AVAILABLE:
            return None, None, None, None

        print("\n🔧 Calculating FLOPs for trained model...")

        try:
            # 创建示例输入 - 使用实际的图像输入格式
            batch_size = 1
            crop_size = self.config.crop_size
            spatial_feature_dim = self.config.get_spatial_feature_dim()

            # 创建图像输入张量
            person_A_images = torch.randn(batch_size, 3, crop_size, crop_size).to(self.device)
            person_B_images = torch.randn(batch_size, 3, crop_size, crop_size).to(self.device)
            spatial_features = torch.randn(batch_size, spatial_feature_dim).to(self.device)

            # 计算FLOPs
            model_for_profile = self.model.module if hasattr(self.model, 'module') else self.model
            flops, params = profile(
                model_for_profile,
                inputs=(person_A_images, person_B_images, spatial_features),
                verbose=False
            )

            # 格式化输出
            flops_str, params_str = clever_format([flops, params], "%.3f")

            print(f"✅ FLOPs calculation completed:")
            print(f"   FLOPs: {flops_str} ({flops:,.0f})")
            print(f"   Params: {params_str} ({params:,.0f})")

            return flops, params, flops_str, params_str

        except Exception as e:
            print(f"❌ FLOPs calculation failed: {e}")
            return None, None, None, None

    def save_results_to_file(self):
        """保存训练结果到JSON文件"""
        # 计算FLOPs
        flops_results = self.calculate_flops()

        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj

        # 确保检查点目录存在
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # 准备保存的数据
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'backbone_name': self.config.backbone_name,
                'visual_feature_dim': self.config.visual_feature_dim,
                'fusion_strategy': self.config.fusion_strategy,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'freeze_blocks': getattr(self.config, 'freeze_blocks', 0),
                'use_geometric': getattr(self.config, 'use_geometric', True),
                'use_scene_context': getattr(self.config, 'use_scene_context', True)
            },
            'hardware_requirements': {
                'flops': flops_results[0] if flops_results and flops_results[0] else None,
                'flops_str': flops_results[2] if flops_results and flops_results[2] else None,
                'parameters': flops_results[1] if flops_results and flops_results[1] else sum(p.numel() for p in self.model.parameters()),
                'params_str': flops_results[3] if flops_results and flops_results[3] else None
            },
            'best_metrics': {
                'best_val_mpca': self.best_val_mpca,
                'best_val_acc': self.best_val_acc
            },
            'epoch_results': convert_numpy_to_list(self.epoch_results)
        }
        
        # 保存到文件
        results_path = os.path.join(self.config.checkpoint_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Training results saved to: {results_path}")
    
    def train(self, train_loader, val_loader, test_loader=None):
        """主训练循环"""
        print(f"\nStarting ResNet Stage2 training...")
        print(f"Epochs: {self.config.epochs}")
        print(f"Early stopping patience: {self.config.early_stopping_patience}")
        print(f"Checkpoint dir: {self.config.checkpoint_dir}")
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{self.config.epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_loss, train_acc, train_mpca, train_details = self.train_epoch(train_loader, epoch)
            
            # 验证
            if epoch % self.config.eval_interval == 0:
                val_loss, val_acc, val_mpca, val_metrics = self.validate_epoch(val_loader, epoch)
                
                # 记录指标
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                self.val_mpcas.append(val_mpca)
                
                # 保存和打印epoch结果（包含验证结果）
                self.save_and_print_epoch_results(
                    epoch, train_loss, train_acc, train_mpca, train_details,
                    val_loss, val_acc, val_mpca, val_metrics
                )
                
                # 学习率调度
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # 检查改进
                metric_improved = False
                if self.config.early_stopping_metric == 'mpca':
                    if val_mpca > self.best_val_mpca:
                        self.best_val_mpca = val_mpca
                        metric_improved = True
                elif self.config.early_stopping_metric == 'accuracy':
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        metric_improved = True
                
                # 保存最佳模型
                if metric_improved:
                    self.epochs_without_improvement = 0
                    if self.config.save_best_only:
                        self.checkpoint_manager.save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            epoch, {'val_accuracy': val_acc, 'val_mpca': val_mpca},
                            'best_model', self.config
                        )
                        print(f"New best model saved! {self.config.early_stopping_metric}: "
                              f"{val_mpca if self.config.early_stopping_metric == 'mpca' else val_acc:.4f}")
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping检查
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {self.config.early_stopping_patience} epochs without improvement")
                    break
                
                # 打印当前最佳结果
                print(f"Best Val MPCA: {self.best_val_mpca:.4f}, Best Val Acc: {self.best_val_acc:.4f}")
                print(f"Epochs without improvement: {self.epochs_without_improvement}")
            else:
                # 只保存和打印训练结果
                self.save_and_print_epoch_results(
                    epoch, train_loss, train_acc, train_mpca, train_details
                )
        
        # 训练完成，保存最终结果
        self.save_results_to_file()
        
        # 训练完成，在测试集上评估
        if test_loader is not None:
            print(f"\n{'='*60}")
            print("FINAL TEST EVALUATION")
            print(f"{'='*60}")

            # 加载最佳模型
            best_model_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                print("Loading best model for test evaluation...")
                self.checkpoint_manager.load_checkpoint(best_model_path, self.model)

            test_loss, test_acc, test_mpca, test_metrics = self.test_epoch(test_loader)

            # 计算FLOPs
            flops_results = self.calculate_flops()
            
            # 转换numpy数组为Python列表以便JSON序列化
            def convert_numpy_to_list(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_list(item) for item in obj]
                else:
                    return obj
            
            serializable_test_metrics = convert_numpy_to_list(test_metrics)
            
            # 保存最终结果
            final_results = {
                'test_accuracy': test_acc,
                'test_mpca': test_mpca,
                'test_metrics': serializable_test_metrics,
                'best_val_accuracy': self.best_val_acc,
                'best_val_mpca': self.best_val_mpca,
                'flops': flops_results[0] if flops_results and flops_results[0] else None,
                'flops_str': flops_results[2] if flops_results and flops_results[2] else None,
                'parameters': flops_results[1] if flops_results and flops_results[1] else None,
                'params_str': flops_results[3] if flops_results and flops_results[3] else None,
                'config': {
                    'backbone_name': self.config.backbone_name,
                    'visual_feature_dim': self.config.visual_feature_dim,
                    'fusion_strategy': self.config.fusion_strategy,
                    'learning_rate': self.config.learning_rate,
                    'batch_size': self.config.batch_size
                }
            }
            
            results_path = os.path.join(self.config.checkpoint_dir, 'final_results.json')
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"\nFinal Results saved to: {results_path}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test MPCA: {test_mpca:.4f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ResNet Stage2 Behavior Classification Training')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='ResNet backbone architecture')
    parser.add_argument('--visual_dim', type=int, default=256,
                       help='Visual feature dimension')
    parser.add_argument('--fusion', type=str, default='concat',
                       choices=['concat', 'bilinear', 'add'],
                       help='Feature fusion strategy')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze ResNet backbone parameters')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='../dataset',
                       help='Path to dataset')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='Frame sampling interval')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Feature control arguments
    parser.add_argument('--use_geometric', action='store_true', default=True,
                       help='Use 7D geometric features in spatial features')
    parser.add_argument('--no_geometric', dest='use_geometric', action='store_false',
                       help='Disable geometric features')
    parser.add_argument('--use_scene_context', action='store_true', default=True,
                       help='Use scene context features in spatial features')
    parser.add_argument('--no_scene_context', dest='use_scene_context', action='store_false',
                       help='Disable scene context features')

    # Other arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/resnet_stage2',
                       help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--freeze_blocks', type=int, default=0,
                       help='Number of early residual blocks to freeze (0-4)')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建配置
    if args.backbone == 'resnet50':
        config = get_resnet50_config()
    elif args.backbone == 'resnet34':
        config = get_resnet18_config()  # Use resnet18 config as base
        config.backbone_name = 'resnet34'
        config.visual_feature_dim = 256  # resnet34 has same output as resnet18
    else:  # resnet18 is default
        config = get_resnet18_config()
    
    # Update config with command line arguments (but keep backbone_name consistent with config choice)
    config.visual_feature_dim = args.visual_dim
    config.fusion_strategy = args.fusion
    config.freeze_backbone = args.freeze_backbone
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.data_path = args.data_path
    config.frame_interval = args.frame_interval
    config.num_workers = args.num_workers
    config.checkpoint_dir = args.checkpoint_dir
    config.log_interval = args.log_interval
    config.freeze_blocks = args.freeze_blocks

    # Update feature control settings
    config.use_geometric = args.use_geometric
    config.use_scene_context = args.use_scene_context
    
    # 打印配置
    config.print_config()
    
    # 创建数据加载器
    print(f"\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_resnet_stage2_data_loaders(config)
    
    # 创建训练器
    trainer = ResNetStage2Trainer(config, device)
    
    # 开始训练
    start_time = time.time()
    trainer.train(train_loader, val_loader, test_loader)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation MPCA: {trainer.best_val_mpca:.4f}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")


if __name__ == '__main__':
    main()