import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse
import os
import time
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from dual_person_classifier import DualPersonInteractionClassifier, DualPersonInteractionLoss
from dual_person_dataset import get_dual_person_data_loaders
from two_stage_classifier import get_class_weights


class DualPersonTrainer:
    """
    Trainer for dual-person interaction classification
    Handles training with individual person feature extraction and fusion
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create dual-person model
        self.model = DualPersonInteractionClassifier(
            backbone_name=config.backbone,
            pretrained=config.pretrained,
            num_interaction_classes=5,
            fusion_method=config.fusion_method,
            shared_backbone=config.shared_backbone
        ).to(self.device)
        
        # Create loss function with class weights
        class_weights = get_class_weights() if config.use_class_weights else None
        self.criterion = DualPersonInteractionLoss(
            stage1_weight=config.stage1_weight,
            stage2_weight=config.stage2_weight,
            class_weights=class_weights,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            feature_regularization=config.feature_regularization,
            reg_weight=config.reg_weight
        ).to(self.device)
        
        # Create optimizer
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        elif config.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(),
                                     lr=config.learning_rate,
                                     momentum=0.9,
                                     weight_decay=config.weight_decay)
        
        # Create scheduler
        if config.scheduler == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=0.1)
        elif config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        else:
            self.scheduler = None
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = {'stage1': [], 'stage2': []}
        self.val_accuracies = {'stage1': [], 'stage2': []}
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Create save directory
        self.save_dir = os.path.join(config.save_dir, 
                                   f"dual_person_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        stage1_correct = 0
        stage2_correct = 0
        stage2_total = 0  # Only count samples with interactions
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            person_A_images = batch['person_A_image'].to(self.device)
            person_B_images = batch['person_B_image'].to(self.device)
            stage1_labels = batch['stage1_label'].to(self.device)
            stage2_labels = batch['stage2_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(person_A_images, person_B_images, stage='both')
            
            # Calculate loss
            loss_dict = self.criterion(outputs, stage1_labels, stage2_labels, stage='both')
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            total_loss += loss.item()
            total_samples += person_A_images.size(0)
            
            # Stage 1 accuracy
            stage1_predictions = torch.argmax(outputs['stage1'], dim=1)
            stage1_correct += (stage1_predictions == stage1_labels).sum().item()
            
            # Stage 2 accuracy (only for samples with interactions)
            interaction_mask = stage1_labels == 1
            if interaction_mask.sum() > 0:
                stage2_outputs_masked = outputs['stage2'][interaction_mask]
                stage2_labels_masked = stage2_labels[interaction_mask]
                stage2_predictions = torch.argmax(stage2_outputs_masked, dim=1)
                stage2_correct += (stage2_predictions == stage2_labels_masked).sum().item()
                stage2_total += stage2_labels_masked.size(0)
            
            # Print progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}, '
                      f'Stage1 Acc: {100. * stage1_correct / total_samples:.2f}%, '
                      f'Stage2 Acc: {100. * stage2_correct / max(stage2_total, 1):.2f}%')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        stage1_acc = stage1_correct / total_samples
        stage2_acc = stage2_correct / max(stage2_total, 1)
        
        return avg_loss, stage1_acc, stage2_acc
    
    def validate(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        stage1_correct = 0
        stage2_correct = 0
        stage2_total = 0
        total_samples = 0
        
        all_stage1_targets = []
        all_stage1_preds = []
        all_stage2_targets = []
        all_stage2_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                person_A_images = batch['person_A_image'].to(self.device)
                person_B_images = batch['person_B_image'].to(self.device)
                stage1_labels = batch['stage1_label'].to(self.device)
                stage2_labels = batch['stage2_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(person_A_images, person_B_images, stage='both')
                
                # Calculate loss
                loss_dict = self.criterion(outputs, stage1_labels, stage2_labels, stage='both')
                total_loss += loss_dict['total_loss'].item()
                
                total_samples += person_A_images.size(0)
                
                # Stage 1 metrics
                stage1_predictions = torch.argmax(outputs['stage1'], dim=1)
                stage1_correct += (stage1_predictions == stage1_labels).sum().item()
                
                all_stage1_targets.extend(stage1_labels.cpu().numpy())
                all_stage1_preds.extend(stage1_predictions.cpu().numpy())
                
                # Stage 2 accuracy (only for samples with interactions)
                interaction_mask = stage1_labels == 1
                if interaction_mask.sum() > 0:
                    stage2_outputs_masked = outputs['stage2'][interaction_mask]
                    stage2_labels_masked = stage2_labels[interaction_mask]
                    stage2_predictions = torch.argmax(stage2_outputs_masked, dim=1)
                    stage2_correct += (stage2_predictions == stage2_labels_masked).sum().item()
                    stage2_total += stage2_labels_masked.size(0)
                    
                    all_stage2_targets.extend(stage2_labels_masked.cpu().numpy())
                    all_stage2_preds.extend(stage2_predictions.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        stage1_acc = stage1_correct / total_samples
        stage2_acc = stage2_correct / max(stage2_total, 1)
        
        # Print detailed metrics every few epochs
        if epoch % 5 == 0 or epoch == self.config.epochs - 1:
            print(f"\n=== Validation Epoch {epoch} ===")
            print(f"Total samples: {total_samples}")
            print(f"Stage 1 accuracy: {stage1_acc:.4f}")
            print(f"Stage 2 samples: {stage2_total}")
            print(f"Stage 2 accuracy: {stage2_acc:.4f}")
            
            # Stage 1 classification report
            if len(all_stage1_targets) > 0:
                stage1_report = classification_report(all_stage1_targets, all_stage1_preds, 
                                                    target_names=['No Interaction', 'Has Interaction'])
                print("Stage 1 Classification Report:")
                print(stage1_report)
        
        return avg_loss, stage1_acc, stage2_acc, all_stage1_targets, all_stage1_preds, all_stage2_targets, all_stage2_preds
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"Starting dual-person training for {self.config.epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Fusion method: {self.config.fusion_method}")
        print(f"Shared backbone: {self.config.shared_backbone}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss, train_s1_acc, train_s2_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_s1_acc, val_s2_acc, val_s1_targets, val_s1_preds, val_s2_targets, val_s2_preds = self.validate(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies['stage1'].append(train_s1_acc)
            self.train_accuracies['stage2'].append(train_s2_acc)
            self.val_accuracies['stage1'].append(val_s1_acc)
            self.val_accuracies['stage2'].append(val_s2_acc)
            
            epoch_time = time.time() - start_time
            
            print(f'\nEpoch {epoch}:')
            print(f'  Train Loss: {train_loss:.4f}, Stage1 Acc: {train_s1_acc:.4f}, Stage2 Acc: {train_s2_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Stage1 Acc: {val_s1_acc:.4f}, Stage2 Acc: {val_s2_acc:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_loss', epoch, val_loss, val_s1_acc, val_s2_acc)
            
            combined_val_acc = (val_s1_acc + val_s2_acc) / 2
            if combined_val_acc > self.best_val_acc:
                self.best_val_acc = combined_val_acc
                self.save_checkpoint('best_accuracy', epoch, val_loss, val_s1_acc, val_s2_acc)
            
            # Save regular checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, val_loss, val_s1_acc, val_s2_acc)
        
        # Save final model
        self.save_checkpoint('final', self.config.epochs, val_loss, val_s1_acc, val_s2_acc)
        
        # Generate final report
        self.generate_final_report(val_s1_targets, val_s1_preds, val_s2_targets, val_s2_preds)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
    
    def save_checkpoint(self, name, epoch, val_loss, val_s1_acc, val_s2_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_stage1_accuracy': val_s1_acc,
            'val_stage2_accuracy': val_s2_acc,
            'config': vars(self.config),
            'training_stats': {
                'total_epochs': len(self.train_losses),
            }
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'{name}.pth'))
        print(f'Checkpoint saved: {name}.pth')
    
    def generate_final_report(self, val_s1_targets, val_s1_preds, val_s2_targets, val_s2_preds):
        """Generate comprehensive evaluation report"""
        print("\nGenerating dual-person evaluation report...")
        
        # Save text report
        with open(os.path.join(self.save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Dual-Person Interaction Classification Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Model configuration
            f.write(f"Model Configuration:\n")
            f.write(f"  Backbone: {self.config.backbone}\n")
            f.write(f"  Fusion method: {self.config.fusion_method}\n")
            f.write(f"  Shared backbone: {self.config.shared_backbone}\n")
            f.write(f"  Feature regularization: {self.config.feature_regularization}\n\n")
            
            # Stage 1 report
            if len(val_s1_targets) > 0:
                stage1_report = classification_report(val_s1_targets, val_s1_preds, 
                                                    target_names=['No Interaction', 'Has Interaction'])
                f.write("Stage 1 (Binary Interaction Detection):\n")
                f.write(stage1_report)
                f.write("\n")
            
            # Stage 2 report
            if len(val_s2_targets) > 0:
                interaction_labels = ['walking_together', 'standing_together', 'conversation', 'sitting_together', 'others']
                stage2_report = classification_report(val_s2_targets, val_s2_preds, 
                                                    target_names=interaction_labels)
                f.write("Stage 2 (Interaction Type Classification):\n")
                f.write(stage2_report)
                f.write("\n")
            
            f.write(f"Best validation loss: {self.best_val_loss:.4f}\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.4f}\n")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Plot confusion matrices
        if len(val_s1_targets) > 0:
            self.plot_confusion_matrix(val_s1_targets, val_s1_preds, 
                                     ['No Interaction', 'Has Interaction'], 'stage1')
        
        if len(val_s2_targets) > 0:
            interaction_labels = ['walking_together', 'standing_together', 'conversation', 'sitting_together', 'others']
            self.plot_confusion_matrix(val_s2_targets, val_s2_preds, interaction_labels, 'stage2')
        
        print(f"Evaluation report saved to: {self.save_dir}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_title(f'Training and Validation Loss\n(Fusion: {self.config.fusion_method})')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Stage 1 accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies['stage1'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.val_accuracies['stage1'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Stage 1 Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Stage 2 accuracy curves
        axes[1, 0].plot(epochs, self.train_accuracies['stage2'], 'b-', label='Training Accuracy')
        axes[1, 0].plot(epochs, self.val_accuracies['stage2'], 'r-', label='Validation Accuracy')
        axes[1, 0].set_title('Stage 2 Accuracy')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined accuracy
        combined_train_acc = [(s1 + s2) / 2 for s1, s2 in zip(self.train_accuracies['stage1'], self.train_accuracies['stage2'])]
        combined_val_acc = [(s1 + s2) / 2 for s1, s2 in zip(self.val_accuracies['stage1'], self.val_accuracies['stage2'])]
        
        axes[1, 1].plot(epochs, combined_train_acc, 'b-', label='Training Combined')
        axes[1, 1].plot(epochs, combined_val_acc, 'r-', label='Validation Combined')
        axes[1, 1].set_title('Combined Accuracy')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, targets, predictions, class_names, stage):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{stage.capitalize()} Confusion Matrix\n(Fusion: {self.config.fusion_method})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix_{stage}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Dual-Person Interaction Classification Training')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='D:/1data/imagedata',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='Input image size [height, width]')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['mobilenet'], help='Backbone network')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'add', 'subtract', 'multiply', 'attention'],
                        help='Feature fusion method')
    parser.add_argument('--shared_backbone', action='store_true', default=True,
                        help='Share backbone weights between two persons')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'none'], help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    
    # Loss function parameters
    parser.add_argument('--stage1_weight', type=float, default=1.0,
                        help='Weight for stage 1 loss')
    parser.add_argument('--stage2_weight', type=float, default=1.0,
                        help='Weight for stage 2 loss')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--feature_regularization', action='store_true', default=False,
                        help='Enable feature regularization between persons')
    parser.add_argument('--reg_weight', type=float, default=0.01,
                        help='Regularization weight')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint saving interval')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("Dual-Person Interaction Classification Training")
    print("="*70)
    print(f"Fusion method: {args.fusion_method}")
    print(f"Shared backbone: {args.shared_backbone}")
    print(f"Feature regularization: {args.feature_regularization}")
    print("")
    
    # Load data
    print("Loading dual-person dataset...")
    train_loader, val_loader, test_loader, interaction_labels = get_dual_person_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_pose=False
    )
    
    print(f"Data loaded successfully!")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Interaction labels: {interaction_labels}")
    
    # Create trainer
    trainer = DualPersonTrainer(args)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("Dual-person training completed!")
    print(f"Results saved to: {trainer.save_dir}")


if __name__ == '__main__':
    main()