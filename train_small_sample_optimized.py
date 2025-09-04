#!/usr/bin/env python3
"""
Optimized training script for small sample sizes (1000-5000 samples per epoch)
Addresses overfitting issues and improves validation performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import argparse
import os
import sys
import time
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightweight_dual_person_classifier import get_optimized_model_for_sample_size
from twostage_training.dual_person_downsampling_dataset import get_dual_person_downsampling_data_loaders


class SmallSampleOptimizedTrainer:
    """
    Specialized trainer for small sample sizes with overfitting prevention
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Get optimized model for sample size
        self.model, self.recommended_config = get_optimized_model_for_sample_size(
            sample_size=config.train_samples_per_epoch,
            backbone_name=config.backbone
        )
        self.model = self.model.to(self.device)
        
        # Override config with recommended settings
        if config.use_recommended_config:
            print("Using recommended configuration for small sample training:")
            for key, value in self.recommended_config.items():
                if hasattr(config, key):
                    old_value = getattr(config, key)
                    setattr(config, key, value)
                    print(f"  {key}: {old_value} -> {value}")
        
        # Create loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Create optimizer with recommended settings
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(trainable_params, 
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        elif config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(trainable_params,
                                       lr=config.learning_rate,
                                       weight_decay=config.weight_decay)
        elif config.optimizer == 'sgd':
            self.optimizer = optim.SGD(trainable_params,
                                     lr=config.learning_rate,
                                     momentum=0.9,
                                     weight_decay=config.weight_decay)
        
        # Create scheduler with early stopping capability
        if config.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', 
                                             factor=0.5, patience=5, verbose=True)
        elif config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        elif config.scheduler == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=0.1)
        else:
            self.scheduler = None
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Early stopping
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 15)
        
        # Create save directory
        self.save_dir = os.path.join(config.save_dir, 
                                   f"small_sample_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params_count:,}")
        print(f"  Training samples per epoch: {config.train_samples_per_epoch:,}")
        print(f"  Parameters per training sample: {trainable_params_count/config.train_samples_per_epoch:.1f}")
        print(f"  Experiment directory: {self.save_dir}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with regularization"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            person_A_images = batch['person_A_image'].to(self.device)
            person_B_images = batch['person_B_image'].to(self.device)
            stage1_labels = batch['stage1_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(person_A_images, person_B_images)
            stage1_output = outputs['stage1']
            
            # Calculate loss
            loss = self.criterion(stage1_output, stage1_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record metrics
            total_loss += loss.item()
            predictions = torch.argmax(stage1_output, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(stage1_labels.cpu().numpy())
            
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(person_A_images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate with detailed metrics"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in val_loader:
                person_A_images = batch['person_A_image'].to(self.device)
                person_B_images = batch['person_B_image'].to(self.device)
                stage1_labels = batch['stage1_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(person_A_images, person_B_images)
                stage1_output = outputs['stage1']
                
                # Calculate loss
                loss = self.criterion(stage1_output, stage1_labels)
                total_loss += loss.item()
                
                # Record predictions
                probabilities = torch.softmax(stage1_output, dim=1)
                predictions = torch.argmax(stage1_output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(stage1_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1, all_targets, all_predictions
    
    def train(self, train_loader, val_loader):
        """Full training loop with early stopping"""
        print(f"Starting small-sample optimized training...")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_targets, val_preds = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start_time
            
            # Print progress
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            print(f'           Time: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping logic
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_accuracy', epoch, val_loss, val_acc, val_f1)
                improved = True
                print(f'           *** New best validation accuracy: {val_acc:.4f} ***')
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_loss', epoch, val_loss, val_acc, val_f1)
                if not improved:
                    improved = True
            
            if improved:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f'           No improvement for {self.patience_counter} epochs')
                
                if self.patience_counter >= self.early_stopping_patience:
                    print(f'Early stopping triggered after {epoch} epochs!')
                    break
            
            # Save regular checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, val_loss, val_acc, val_f1)
        
        # Generate final report
        self.generate_final_report()
        
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Results saved to: {self.save_dir}")
    
    def save_checkpoint(self, name, epoch, val_loss, val_acc, val_f1):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'best_val_acc': self.best_val_acc,
            'config': vars(self.config)
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'{name}.pth'))
    
    def generate_final_report(self):
        """Generate training report with overfitting analysis"""
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training vs Validation Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.learning_rates, 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Overfitting analysis
        train_val_gap = np.array(self.train_accuracies) - np.array(self.val_accuracies)
        axes[1, 1].plot(epochs, train_val_gap, 'purple', label='Train-Val Gap')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Analysis (Train-Val Gap)')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Accuracy Difference')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        with open(os.path.join(self.save_dir, 'training_report.txt'), 'w') as f:
            f.write("Small Sample Optimized Training Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Training samples per epoch: {self.config.train_samples_per_epoch}\n")
            f.write(f"  Validation samples: {len(self.val_accuracies)} epochs evaluated\n")
            f.write(f"  Model: {type(self.model).__name__}\n")
            f.write(f"  Learning rate: {self.config.learning_rate}\n")
            f.write(f"  Weight decay: {self.config.weight_decay}\n")
            f.write(f"  Early stopping patience: {self.early_stopping_patience}\n\n")
            
            f.write(f"Results:\n")
            f.write(f"  Best validation accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"  Final training accuracy: {self.train_accuracies[-1]:.4f}\n")
            f.write(f"  Final validation accuracy: {self.val_accuracies[-1]:.4f}\n")
            f.write(f"  Training epochs: {len(self.train_accuracies)}\n")
            
            # Overfitting analysis
            final_gap = self.train_accuracies[-1] - self.val_accuracies[-1]
            max_gap = max(np.array(self.train_accuracies) - np.array(self.val_accuracies))
            f.write(f"  Final train-val gap: {final_gap:.4f}\n")
            f.write(f"  Maximum train-val gap: {max_gap:.4f}\n")
            
            if max_gap > 0.1:
                f.write("  ⚠️ Significant overfitting detected\n")
            elif max_gap > 0.05:
                f.write("  ⚠️ Mild overfitting detected\n")
            else:
                f.write("  ✅ No significant overfitting\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Small Sample Optimized Training')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='D:/1data/imagedata',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='Input image size [height, width]')
    
    # Sampling parameters
    parser.add_argument('--train_samples_per_epoch', type=int, default=1000,
                        help='Number of training samples per epoch')
    parser.add_argument('--val_samples_per_epoch', type=int, default=300,
                        help='Number of validation samples per epoch')
    parser.add_argument('--balance_train_classes', action='store_true', default=True,
                        help='Balance classes in training set')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['mobilenet'], help='Backbone network')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'step', 'cosine', 'none'], help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Configuration
    parser.add_argument('--use_recommended_config', action='store_true', default=True,
                        help='Use recommended configuration for sample size')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint saving interval')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Small Sample Optimized Training")
    print("=" * 80)
    print(f"Training samples per epoch: {args.train_samples_per_epoch}")
    print(f"Validation samples per epoch: {args.val_samples_per_epoch}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, interaction_labels = get_dual_person_downsampling_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_pose=False,
        crop_padding=20,
        min_person_size=32,
        train_samples_per_epoch=args.train_samples_per_epoch,
        balance_train_classes=args.balance_train_classes,
        val_samples_per_epoch=args.val_samples_per_epoch,
        test_samples_per_epoch=None
    )
    
    print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create trainer
    trainer = SmallSampleOptimizedTrainer(args)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()