#!/usr/bin/env python3
"""
Unified Training Script for Stage2 Behavior Classification
Supports both Basic and LSTM modes with modular architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

# 导入模块化组件
from configs.stage2_config import Stage2Config, add_config_args, create_config_from_args
from utils.model_factory import create_full_training_setup, ModelCheckpointManager
from utils.data_factory import create_stage2_data_loaders, print_dataset_summary
from models.stage2_classifier import Stage2Evaluator


class Stage2Trainer:
    """统一的Stage2训练器，支持Basic和LSTM模式"""
    
    def __init__(self, config: Stage2Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join('stage2_experiments', f'{config.model_type}_{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
        print(f"📁 Experiment directory: {self.save_dir}")
        
        # 创建训练组件
        self.model, self.criterion, self.optimizer, self.scheduler = create_full_training_setup(
            config, self.device
        )
        
        # 检查点管理器
        self.checkpoint_manager = ModelCheckpointManager(self.save_dir)
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_mpcas = []
        
        # 最佳模型记录
        self.best_val_mpca = 0.0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # 类别名称
        self.class_names = [
            'Walking Together',   # 移动行为
            'Standing Together',  # 站立行为
            'Sitting Together'    # 坐着行为
        ]
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float, Dict]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_ce_loss = 0
        # total_mpca_loss = 0  # 已注释MPCA损失
        total_acc = 0
        
        evaluator = Stage2Evaluator(self.class_names)
        
        for batch_idx, batch in enumerate(train_loader):
            # 处理不同模式的输入数据格式
            targets = batch['stage2_label'].to(self.device)
            
            if self.config.temporal_mode == 'lstm':
                # LSTM模式使用sequences字段 [batch_size, seq_len, feat_dim]
                inputs = batch['sequences'].to(self.device)
                outputs = self.model(inputs)
            elif self.config.temporal_mode == 'relation':
                # Relation模式使用分离的特征
                person_A_features = batch['person_A_features'].to(self.device)
                person_B_features = batch['person_B_features'].to(self.device)
                spatial_features = batch['spatial_features'].to(self.device)
                outputs = self.model(person_A_features, person_B_features, spatial_features)
            else:
                # Basic模式使用features字段 [batch_size, feat_dim]
                inputs = batch['features'].to(self.device)
                outputs = self.model(inputs)
            loss, loss_dict = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            
            self.optimizer.step()
            
            # 记录指标
            total_loss += loss.item()
            total_ce_loss += loss_dict['ce_loss']
            # total_mpca_loss += loss_dict['mpca_loss']  # 已注释MPCA损失
            total_acc += loss_dict['overall_acc']
            
            # 更新评估器
            evaluator.update(outputs, targets)
            
            # 打印进度
            if batch_idx % self.config.log_interval == 0:
                progress = 100. * batch_idx / len(train_loader)
                batch_size = targets.size(0)  # 使用targets获取批次大小，所有模式都有
                print(f'Train Epoch {epoch}: [{batch_idx * batch_size}/{len(train_loader.dataset)} '
                      f'({progress:.0f}%)] Loss: {loss.item():.6f}, Acc: {loss_dict["overall_acc"]:.4f}')
        
        # 计算平均指标
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        # avg_mpca_loss = total_mpca_loss / num_batches  # 已注释MPCA损失
        avg_acc = total_acc / num_batches
        
        # 计算详细评估指标
        metrics = evaluator.compute_metrics()
        train_mpca = metrics.get('mpca', 0.0)
        
        return avg_loss, avg_acc, train_mpca, {
            'ce_loss': avg_ce_loss,
            # 'mpca_loss': avg_mpca_loss,  # 已注释MPCA损失
            'detailed_metrics': metrics
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float, float, Dict]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0
        evaluator = Stage2Evaluator(self.class_names)
        
        with torch.no_grad():
            for batch in val_loader:
                # 处理不同模式的输入数据格式
                targets = batch['stage2_label'].to(self.device)
                
                if self.config.temporal_mode == 'lstm':
                    # LSTM模式使用sequences字段
                    inputs = batch['sequences'].to(self.device)
                    outputs = self.model(inputs)
                elif self.config.temporal_mode == 'relation':
                    # Relation模式使用分离的特征
                    person_A_features = batch['person_A_features'].to(self.device)
                    person_B_features = batch['person_B_features'].to(self.device)
                    spatial_features = batch['spatial_features'].to(self.device)
                    outputs = self.model(person_A_features, person_B_features, spatial_features)
                else:
                    # Basic模式使用features字段
                    inputs = batch['features'].to(self.device)
                    outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                
                # 记录指标
                total_loss += loss.item()
                
                # 更新评估器
                evaluator.update(outputs, targets)
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算详细评估指标
        metrics = evaluator.compute_metrics()
        val_acc = metrics.get('overall_accuracy', 0.0)
        val_mpca = metrics.get('mpca', 0.0)
        
        return avg_loss, val_acc, val_mpca, metrics
    
    def test_epoch(self, test_loader: DataLoader) -> Tuple[float, float, float, Dict]:
        """测试评估"""
        self.model.eval()
        
        total_loss = 0
        evaluator = Stage2Evaluator(self.class_names)
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                targets = batch['stage2_label'].to(self.device)
                
                # 前向传播
                outputs = self.model(features)
                loss, loss_dict = self.criterion(outputs, targets)
                
                # 记录指标
                total_loss += loss.item()
                
                # 更新评估器
                evaluator.update(outputs, targets)
        
        avg_loss = total_loss / len(test_loader)
        
        # 计算详细评估指标
        metrics = evaluator.compute_metrics()
        test_acc = metrics.get('overall_accuracy', 0.0)
        test_mpca = metrics.get('mpca', 0.0)
        
        return avg_loss, test_acc, test_mpca, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              test_loader: Optional[DataLoader] = None):
        """主训练循环"""
        print(f"\n🚀 Starting Stage2 training for {self.config.epochs} epochs...")
        print(f"Model: {self.config.model_type}")
        print(f"Input dimension: {self.config.get_input_dim()}D")
        print(f"Training samples: {len(train_loader.dataset):,}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc, train_mpca, train_details = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc, val_mpca, val_metrics = self.validate_epoch(val_loader, epoch)
            
            # 记录指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.val_mpcas.append(val_mpca)
            
            # 学习率调度
            if self.scheduler:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 检查改进
            if self.config.early_stopping_metric == 'mpca':
                improved = val_mpca > self.best_val_mpca
            elif self.config.early_stopping_metric == 'accuracy':
                improved = val_acc > self.best_val_acc
            else:  # loss
                improved = val_loss < self.best_val_loss
            
            if improved:
                self.best_val_mpca = val_mpca
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # 保存最佳模型
                metrics = {
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_mpca': val_mpca
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, metrics, 'best_model', self.config
                )
            else:
                self.epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印epoch结果
            print(f'\nEpoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train MPCA: {train_mpca:.4f}')
            print(f'          Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MPCA: {val_mpca:.4f}')
            print(f'          LR: {current_lr:.6f}, Time: {epoch_time:.1f}s')
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f'\n⏹️  Early stopping triggered after {epoch} epochs '
                      f'(no {self.config.early_stopping_metric} improvement for {self.config.early_stopping_patience} epochs)')
                break
            
            # 保存周期性检查点
            if epoch % 10 == 0:
                metrics = {
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_mpca': val_mpca
                }
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, metrics, f'epoch_{epoch}', self.config
                )
        
        total_time = time.time() - start_time
        print(f'\n✅ Stage2 training completed in {total_time/60:.1f} minutes')
        print(f'Best validation MPCA: {self.best_val_mpca:.4f}')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        print(f'Best validation loss: {self.best_val_loss:.4f}')
        
        # 生成训练报告
        self.generate_final_report(val_metrics)
        self.plot_training_curves()
        
        # 测试评估
        if test_loader is not None:
            print("\n🔍 Evaluating on test set...")
            test_loss, test_acc, test_mpca, test_metrics = self.test_epoch(test_loader)
            
            print(f'Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, MPCA: {test_mpca:.4f}')
            
            # 生成测试报告
            self.generate_test_report(test_metrics, test_loss, test_acc, test_mpca)
    
    def generate_final_report(self, val_metrics: Dict):
        """生成最终评估报告"""
        print("\n📊 Generating evaluation report...")
        
        report_path = os.path.join(self.save_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("Stage2 Behavior Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.config.model_type}\n")
            f.write(f"Input dimension: {self.config.get_input_dim()}D\n")
            f.write(f"Classes: 3 basic behavior categories\n")
            f.write(f"Best Validation MPCA: {self.best_val_mpca:.4f}\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n\n")
            
            # 特征配置
            f.write("Feature Configuration:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Geometric features: {self.config.use_geometric}\n")
            f.write(f"HoG features: {self.config.use_hog}\n")
            f.write(f"Scene context: {self.config.use_scene_context}\n")
            f.write(f"Temporal mode: {self.config.temporal_mode}\n")
            f.write(f"Frame interval: {self.config.frame_interval}\n\n")
            
            # 详细分类报告
            if 'class_names' in val_metrics:
                f.write("Per-Class Performance:\n")
                f.write("-" * 30 + "\n")
                for i, class_name in enumerate(val_metrics['class_names']):
                    if i < len(val_metrics['precision']):
                        f.write(f"{class_name}: Precision={val_metrics['precision'][i]:.4f}, "
                               f"Recall={val_metrics['recall'][i]:.4f}, "
                               f"F1={val_metrics['f1_score'][i]:.4f}\n")
            
            f.write(f"\nConfusion Matrix:\n{val_metrics.get('confusion_matrix', 'N/A')}\n")
        
        print(f"📝 Evaluation report saved to {report_path}")
    
    def generate_test_report(self, test_metrics: Dict, test_loss: float, 
                            test_acc: float, test_mpca: float):
        """生成测试报告"""
        report_path = os.path.join(self.save_dir, 'test_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("Stage2 Test Set Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test MPCA: {test_mpca:.4f}\n\n")
            
            # 详细分类报告
            if 'class_names' in test_metrics:
                f.write("Test Per-Class Performance:\n")
                f.write("-" * 30 + "\n")
                for i, class_name in enumerate(test_metrics['class_names']):
                    if i < len(test_metrics['precision']):
                        f.write(f"{class_name}: Precision={test_metrics['precision'][i]:.4f}, "
                               f"Recall={test_metrics['recall'][i]:.4f}, "
                               f"F1={test_metrics['f1_score'][i]:.4f}\n")
            
            f.write(f"\nTest Confusion Matrix:\n{test_metrics.get('confusion_matrix', 'N/A')}\n")
        
        print(f"📝 Test evaluation report saved to {report_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.train_losses) == 0:
            print("⚠️  No training data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', alpha=0.7)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', alpha=0.7)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy', alpha=0.7)
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Val Accuracy', alpha=0.7)
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MPCA曲线
        axes[1, 0].plot(epochs, self.val_mpcas, 'g-', label='Val MPCA', alpha=0.7)
        axes[1, 0].set_title('Validation MPCA')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MPCA')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 综合对比
        axes[1, 1].plot(epochs, self.val_accuracies, 'r-', label='Val Accuracy', alpha=0.7)
        axes[1, 1].plot(epochs, self.val_mpcas, 'g-', label='Val MPCA', alpha=0.7)
        axes[1, 1].set_title('Validation Accuracy vs MPCA')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves saved to {plot_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train Stage2 Behavior Classifier')
    
    # 添加配置参数
    add_config_args(parser)
    
    args = parser.parse_args()
    
    # 创建配置
    config = create_config_from_args(args)
    
    print("🔧 Stage2 Training Configuration:")
    print(f"  Mode: {config.temporal_mode}")
    print(f"  Features: Geometric={config.use_geometric}, HoG={config.use_hog}, Scene={config.use_scene_context}")
    print(f"  Input dimension: {config.get_input_dim()}D")
    print(f"  Frame interval: {config.frame_interval}")
    print(f"  Batch size: {config.batch_size}")
    
    # 创建数据加载器
    print(f"\n📊 Creating data loaders...")
    train_loader, val_loader, test_loader = create_stage2_data_loaders(config)
    
    # 打印数据集摘要
    print_dataset_summary(config, (train_loader, val_loader, test_loader))
    
    # 创建训练器
    trainer = Stage2Trainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader, test_loader)
    
    print(f"\n🎉 Stage2 training completed!")
    print(f"📁 Results saved to: {trainer.save_dir}")


if __name__ == '__main__':
    main()