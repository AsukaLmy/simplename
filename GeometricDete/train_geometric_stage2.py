#!/usr/bin/env python3
"""
Training script for Stage2 geometric behavior classification
5-class behavior classification using 16D geometric+temporal features
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from optimized_stage2_data_loader import create_fast_stage2_data_loaders
from geometric_stage2_classifier import GeometricStage2Classifier, Stage2Loss, Stage2Evaluator


class GeometricStage2Trainer:
    """Stage2行为分类训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join('stage2_experiments', f'stage2_{config.model_type}_{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        # 初始化模型
        self.model = GeometricStage2Classifier(
            input_dim=16,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            use_attention=config.use_attention
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 类别权重
        class_weights = {0: 1.0, 1: 1.4, 2: 8.3, 3: 7.4, 4: 50.0}
        
        # 损失函数
        self.criterion = Stage2Loss(
            class_weights=class_weights,
            mpca_weight=config.mpca_weight,
            acc_weight=config.acc_weight
        )
        
        # 优化器
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        
        # 学习率调度器
        if config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.step_size, gamma=0.1
            )
        elif config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        elif config.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_mpcas = []
        
        # 最佳模型记录
        self.best_val_mpca = 0.0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # 类别名称
        self.class_names = [
            'Static Group', 'Parallel Movement', 'Approaching Interaction',
            'Coordinated Activity', 'Complex/Rare Behaviors'
        ]
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_ce_loss = 0
        total_mpca_loss = 0
        total_acc = 0
        
        evaluator = Stage2Evaluator(self.class_names)
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(self.device)
            targets = batch['stage2_label'].to(self.device)
            
            # 前向传播
            outputs = self.model(features)
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
            total_mpca_loss += loss_dict['mpca_loss']
            total_acc += loss_dict['overall_acc']
            
            # 更新评估器
            evaluator.update(outputs, targets)
            
            # 打印进度
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch {epoch}: [{batch_idx * len(features)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)] '
                      f'Loss: {loss.item():.6f}, Acc: {loss_dict["overall_acc"]:.4f}')
        
        # 计算平均指标
        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_mpca_loss = total_mpca_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        
        # 计算详细评估指标
        metrics = evaluator.compute_metrics()
        train_mpca = metrics.get('mpca', 0.0)
        
        return avg_loss, avg_acc, train_mpca, {
            'ce_loss': avg_ce_loss,
            'mpca_loss': avg_mpca_loss,
            'detailed_metrics': metrics
        }
    
    def validate_epoch(self, val_loader, epoch):
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0
        evaluator = Stage2Evaluator(self.class_names)
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = batch['stage2_label'].to(self.device)
                
                # 前向传播
                outputs = self.model(features)
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
    
    def test_epoch(self, test_loader):
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
    
    def train(self, train_loader, val_loader, test_loader=None):
        """主训练循环"""
        print(f"Starting Stage2 training for {self.config.epochs} epochs...")
        
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
            
            # 检查改进（以MPCA为主要指标）
            improved = val_mpca > self.best_val_mpca
            if improved:
                self.best_val_mpca = val_mpca
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                
                # 保存最佳模型
                self.save_checkpoint('best_model', epoch, val_loss, val_acc, val_mpca)
            else:
                self.epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train MPCA: {train_mpca:.4f}')
            print(f'          Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MPCA: {val_mpca:.4f}')
            print(f'          LR: {current_lr:.6f}, Time: {epoch_time:.1f}s')
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs (no MPCA improvement for {self.config.early_stopping_patience} epochs)')
                break
            
            # 保存周期性检查点
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, val_loss, val_acc, val_mpca)
        
        total_time = time.time() - start_time
        print(f'\nStage2 training completed in {total_time:.1f} seconds')
        print(f'Best validation MPCA: {self.best_val_mpca:.4f}')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        
        # 生成最终报告
        self.generate_final_report(val_metrics)
        self.plot_training_curves()
        
        # 测试评估
        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_loss, test_acc, test_mpca, test_metrics = self.test_epoch(test_loader)
            
            print(f'Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, MPCA: {test_mpca:.4f}')
            
            # 生成测试报告
            self.generate_test_report(test_metrics, test_loss, test_acc, test_mpca)
    
    def save_checkpoint(self, name, epoch, val_loss, val_acc, val_mpca):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_mpca': val_mpca,
            'config': vars(self.config)
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'{name}.pth'))
        print(f'Checkpoint saved: {name}.pth')
    
    def generate_final_report(self, val_metrics):
        """生成最终评估报告"""
        print("\nGenerating Stage2 evaluation report...")
        
        report_path = os.path.join(self.save_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("Stage2 Behavior Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: GeometricStage2Classifier\n")
            f.write(f"Features: 16D (7 geometric + 4 basic motion + 5 enhanced temporal)\n")
            f.write(f"Classes: 5 behavior categories\n")
            f.write(f"Best Validation MPCA: {self.best_val_mpca:.4f}\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.4f}\n\n")
            
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
        
        print(f"Evaluation report saved to {report_path}")
    
    def generate_test_report(self, test_metrics, test_loss, test_acc, test_mpca):
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
        
        print(f"Test evaluation report saved to {report_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy')
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MPCA曲线
        axes[1, 0].plot(epochs, self.val_mpcas, 'g-', label='Val MPCA')
        axes[1, 0].set_title('Validation MPCA')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MPCA')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 综合对比
        axes[1, 1].plot(epochs, self.val_accuracies, 'r-', label='Val Accuracy')
        axes[1, 1].plot(epochs, self.val_mpcas, 'g-', label='Val MPCA')
        axes[1, 1].set_title('Validation Accuracy vs MPCA')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {os.path.join(self.save_dir, 'training_curves.png')}")


# Removed: Now using create_fast_stage2_data_loaders from optimized_stage2_data_loader.py


def main():
    parser = argparse.ArgumentParser(description='Train Stage2 Geometric Behavior Classifier')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='geometric_stage2',
                        help='Model type identifier')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32, 16],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use attention mechanism')
    parser.add_argument('--history_length', type=int, default=5,
                        help='Length of temporal history')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for step scheduler')
    
    # 损失函数参数
    parser.add_argument('--mpca_weight', type=float, default=0.1,
                        help='MPCA regularization weight')
    parser.add_argument('--acc_weight', type=float, default=0.05,
                        help='Accuracy regularization weight')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # 训练控制
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    
    # 特征选项
    parser.add_argument('--use_temporal', action='store_true', default=True,
                        help='Use temporal features')
    parser.add_argument('--use_scene_context', action='store_true', default=True,
                        help='Use scene context features')
    
    args = parser.parse_args()
    
    # 创建数据加载器
    print("Loading Stage2 geometric data...")
    train_loader, val_loader, test_loader = create_fast_stage2_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history_length=args.history_length,
        use_temporal=args.use_temporal,
        use_scene_context=args.use_scene_context
    )
    
    # 创建训练器
    trainer = GeometricStage2Trainer(args)
    
    # 开始训练
    trainer.train(train_loader, val_loader, test_loader)
    
    print("Stage2 training completed!")


if __name__ == '__main__':
    main()