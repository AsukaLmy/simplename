#!/usr/bin/env python3
"""
Train Visual-Enhanced Stage1 Classifier
Compare visual+geometric vs geometric-only for Stage1 interaction detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from datetime import datetime
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from visual_geometric_stage1_classifier import (
    VisualGeometricStage1Classifier, 
    VisualOnlyStage1Classifier,
    HybridStage1Ensemble
)
from visual_stage1_dataset import create_visual_stage1_data_loaders


class VisualStage1Trainer:
    """Trainer for visual-enhanced Stage1 interaction detection"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('checkpoints', f'visual_stage1_{config.model_type}_{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
        
        # Setup training
        self._setup_training()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        # Save config
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        print(f"VisualStage1Trainer initialized on {self.device}")
        print(f"Save directory: {self.save_dir}")
    
    def _initialize_model(self):
        """Initialize model based on config"""
        if self.config.model_type == 'visual_geometric':
            self.model = VisualGeometricStage1Classifier(
                backbone_name=self.config.backbone,
                visual_feature_dim=self.config.visual_feature_dim,
                fusion_strategy=self.config.fusion_strategy,
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
                pretrained=self.config.pretrained,
                freeze_backbone=self.config.freeze_backbone
            )
        elif self.config.model_type == 'visual_only':
            self.model = VisualOnlyStage1Classifier(
                backbone_name=self.config.backbone,
                visual_feature_dim=self.config.visual_feature_dim,
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
                pretrained=self.config.pretrained,
                freeze_backbone=self.config.freeze_backbone
            )
        elif self.config.model_type == 'ensemble':
            self.model = HybridStage1Ensemble(
                backbone_name=self.config.backbone,
                visual_feature_dim=self.config.visual_feature_dim
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {self.config.model_type}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if hasattr(self.model, 'get_feature_info'):
            print(f"Feature info: {self.model.get_feature_info()}")
    
    def _setup_training(self):
        """Setup optimizer, criterion, scheduler"""
        # Loss with class weighting if specified
        if hasattr(self.config, 'class_weights') and self.config.class_weights:
            weights = torch.tensor([self.config.class_weights.get(0, 1.0), 
                                   self.config.class_weights.get(1, 1.0)]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"Using class weights: {self.config.class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        
        # Scheduler
        if self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.step_size, gamma=0.5
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            person_A_img = batch['person_A_image'].to(self.device)
            person_B_img = batch['person_B_image'].to(self.device)
            geometric_features = batch['geometric_features'].to(self.device)
            targets = batch['stage1_label'].to(self.device)
            
            # Forward pass
            if self.config.model_type == 'visual_only':
                outputs = self.model(person_A_img, person_B_img)
            else:
                outputs = self.model(person_A_img, person_B_img, geometric_features)
            
            # Loss
            loss = self.criterion(outputs, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch {epoch}: [{batch_idx * len(person_A_img)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, val_loader, epoch):
        """Validate one epoch"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                person_A_img = batch['person_A_image'].to(self.device)
                person_B_img = batch['person_B_image'].to(self.device)
                geometric_features = batch['geometric_features'].to(self.device)
                targets = batch['stage1_label'].to(self.device)
                
                # Forward pass
                if self.config.model_type == 'visual_only':
                    outputs = self.model(person_A_img, person_B_img)
                else:
                    outputs = self.model(person_A_img, person_B_img, geometric_features)
                
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1, all_targets, all_predictions
    
    def train(self, train_loader, val_loader, test_loader=None):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_targets, val_predictions = self.validate_epoch(val_loader, epoch)
            
            # Record
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Scheduler
            if self.scheduler:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check improvement
            improved = val_acc > self.best_val_acc
            if improved:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint('best_model', epoch, val_loss, val_acc, val_f1)
            else:
                self.epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, '
                  f'Time: {epoch_time:.1f}s')
            
            # Early stopping
            if hasattr(self.config, 'early_stopping_patience') and \
               self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
            
            # Periodic save
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, val_loss, val_acc, val_f1)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.1f} seconds')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        print(f'Best validation F1: {self.best_val_f1:.4f}')
        
        # Generate reports
        self.generate_report(val_targets, val_predictions)
        self.plot_training_curves()
        
        # Test evaluation
        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_loss, test_acc, test_f1, test_targets, test_predictions = self.validate_epoch(test_loader, -1)
            print(f'Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}')
            self.generate_test_report(test_targets, test_predictions, test_loss, test_acc, test_f1)
    
    def save_checkpoint(self, name, epoch, val_loss, val_acc, val_f1):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'config': vars(self.config)
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'{name}.pth'))
        print(f'Checkpoint saved: {name}.pth')
    
    def generate_report(self, val_targets, val_predictions):
        """Generate evaluation report"""
        class_names = ['No Interaction', 'Has Interaction']
        report = classification_report(val_targets, val_predictions, target_names=class_names)
        cm = confusion_matrix(val_targets, val_predictions)
        
        with open(os.path.join(self.save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Visual Stage1 Binary Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Backbone: {self.config.backbone}\n")
            f.write(f"Fusion Strategy: {getattr(self.config, 'fusion_strategy', 'N/A')}\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"Best Validation F1: {self.best_val_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write(f"\n\nConfusion Matrix:\n{cm}\n")
        
        print(f"Report saved to {self.save_dir}")
    
    def generate_test_report(self, test_targets, test_predictions, test_loss, test_acc, test_f1):
        """Generate test report"""
        class_names = ['No Interaction', 'Has Interaction']
        report = classification_report(test_targets, test_predictions, target_names=class_names)
        cm = confusion_matrix(test_targets, test_predictions)
        
        with open(os.path.join(self.save_dir, 'test_evaluation_report.txt'), 'w') as f:
            f.write("Visual Stage1 Test Set Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1 Score: {test_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write(f"\n\nConfusion Matrix:\n{cm}\n")
        
        print(f"Test report saved")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Visual Stage1 Classifier')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size for person images')
    parser.add_argument('--frame_interval', type=int, default=5, help='Frame sampling interval')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='visual_geometric',
                       choices=['visual_geometric', 'visual_only', 'ensemble'],
                       help='Type of model')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='ResNet backbone')
    parser.add_argument('--visual_feature_dim', type=int, default=256, help='Visual feature dimension')
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                       choices=['concat', 'add', 'attention'],
                       help='Feature fusion strategy')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                       help='Hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained backbone')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['step', 'cosine', 'plateau', 'none'], help='Scheduler')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for step scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print(f"Visual Stage1 Training Configuration:")
    print(f"  Model: {args.model_type}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Data path: {args.data_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_visual_stage1_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        backbone_name=args.backbone,
        crop_size=args.crop_size,
        frame_interval=args.frame_interval
    )
    
    # Create trainer
    trainer = VisualStage1Trainer(args)
    
    # Train
    trainer.train(train_loader, val_loader, test_loader)
    
    print("Training completed!")


if __name__ == '__main__':
    main()