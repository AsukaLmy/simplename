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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from two_stage_classifier import TwoStageInteractionClassifier, InteractionLoss, get_class_weights
from dataset import get_data_loaders


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = TwoStageInteractionClassifier(
            backbone_name=config.backbone,
            pretrained=config.pretrained,
            num_interaction_classes=5
        ).to(self.device)
        
        # Create loss function with class weights
        class_weights = get_class_weights() if config.use_class_weights else None
        self.criterion = InteractionLoss(
            stage1_weight=config.stage1_weight,
            stage2_weight=config.stage2_weight,
            class_weights=class_weights,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
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
                                   f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
            images = batch['image'].to(self.device)
            stage1_labels = batch['stage1_label'].to(self.device)
            stage2_labels = batch['stage2_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, stage='both')
            
            # Calculate loss
            loss_dict = self.criterion(outputs, stage1_labels, stage2_labels, stage='both')
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_samples += images.size(0)
            
            # Stage 1 accuracy
            stage1_preds = torch.argmax(outputs['stage1'], dim=1)
            stage1_correct += (stage1_preds == stage1_labels).sum().item()
            
            # Stage 2 accuracy (only for samples with interactions)
            interaction_mask = stage1_labels == 1
            if interaction_mask.sum() > 0:
                stage2_outputs_filtered = outputs['stage2'][interaction_mask]
                stage2_labels_filtered = stage2_labels[interaction_mask]
                stage2_preds = torch.argmax(stage2_outputs_filtered, dim=1)
                stage2_correct += (stage2_preds == stage2_labels_filtered).sum().item()
                stage2_total += stage2_labels_filtered.size(0)
            
            # Print progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Stage1 Acc: {100. * stage1_correct / total_samples:.2f}%, '
                      f'Stage2 Acc: {100. * stage2_correct / max(stage2_total, 1):.2f}%')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        stage1_acc = stage1_correct / total_samples
        stage2_acc = stage2_correct / max(stage2_total, 1)
        
        self.train_losses.append(avg_loss)
        self.train_accuracies['stage1'].append(stage1_acc)
        self.train_accuracies['stage2'].append(stage2_acc)
        
        return avg_loss, stage1_acc, stage2_acc
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        stage1_correct = 0
        stage2_correct = 0
        stage2_total = 0
        total_samples = 0
        
        all_stage1_preds = []
        all_stage1_targets = []
        all_stage2_preds = []
        all_stage2_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                stage1_labels = batch['stage1_label'].to(self.device)
                stage2_labels = batch['stage2_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, stage='both')
                
                # Calculate loss
                loss_dict = self.criterion(outputs, stage1_labels, stage2_labels, stage='both')
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                total_samples += images.size(0)
                
                # Stage 1 predictions and accuracy
                stage1_preds = torch.argmax(outputs['stage1'], dim=1)
                stage1_correct += (stage1_preds == stage1_labels).sum().item()
                
                all_stage1_preds.extend(stage1_preds.cpu().numpy())
                all_stage1_targets.extend(stage1_labels.cpu().numpy())
                
                # Stage 2 accuracy (only for samples with interactions)
                interaction_mask = stage1_labels == 1
                if interaction_mask.sum() > 0:
                    stage2_outputs_filtered = outputs['stage2'][interaction_mask]
                    stage2_labels_filtered = stage2_labels[interaction_mask]
                    stage2_preds = torch.argmax(stage2_outputs_filtered, dim=1)
                    stage2_correct += (stage2_preds == stage2_labels_filtered).sum().item()
                    stage2_total += stage2_labels_filtered.size(0)
                    
                    all_stage2_preds.extend(stage2_preds.cpu().numpy())
                    all_stage2_targets.extend(stage2_labels_filtered.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        stage1_acc = stage1_correct / total_samples
        stage2_acc = stage2_correct / max(stage2_total, 1)
        
        self.val_losses.append(avg_loss)
        self.val_accuracies['stage1'].append(stage1_acc)
        self.val_accuracies['stage2'].append(stage2_acc)
        
        # Print detailed metrics every few epochs
        if epoch % 5 == 0 or epoch == self.config.epochs - 1:
            print(f"\n=== Validation Epoch {epoch} ===")
            print("Stage 1 (Interaction Detection):")
            print(classification_report(all_stage1_targets, all_stage1_preds, 
                                      target_names=['No Interaction', 'Has Interaction']))
            
            if len(all_stage2_targets) > 0:
                print("Stage 2 (Interaction Type Classification):")
                print(classification_report(all_stage2_targets, all_stage2_preds))
            
            print("=" * 50)
        
        return avg_loss, stage1_acc, stage2_acc, all_stage1_targets, all_stage1_preds, all_stage2_targets, all_stage2_preds
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_s1_acc, train_s2_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_s1_acc, val_s2_acc, val_s1_targets, val_s1_preds, val_s2_targets, val_s2_preds = self.validate(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f'\nEpoch {epoch}:')
            print(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'  Train S1 Acc: {train_s1_acc:.4f}, Val S1 Acc: {val_s1_acc:.4f}')
            print(f'  Train S2 Acc: {train_s2_acc:.4f}, Val S2 Acc: {val_s2_acc:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_loss_model.pth')
                print("  -> Saved best loss model")
            
            # Combined accuracy for best model selection
            combined_acc = 0.7 * val_s1_acc + 0.3 * val_s2_acc
            if combined_acc > self.best_val_acc:
                self.best_val_acc = combined_acc
                self.save_model('best_acc_model.pth')
                print("  -> Saved best accuracy model")
            
            # Save training curves
            self.plot_training_curves()
        
        print("Training completed!")
        
        # Save final model
        self.save_model('final_model.pth')
        
        # Generate final evaluation
        self.generate_final_report(val_s1_targets, val_s1_preds, val_s2_targets, val_s2_preds)
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Stage 1 accuracy
        ax2.plot(self.train_accuracies['stage1'], label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies['stage1'], label='Val Acc', color='red')
        ax2.set_title('Stage 1 Accuracy (Interaction Detection)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Stage 2 accuracy
        ax3.plot(self.train_accuracies['stage2'], label='Train Acc', color='blue')
        ax3.plot(self.val_accuracies['stage2'], label='Val Acc', color='red')
        ax3.set_title('Stage 2 Accuracy (Interaction Type)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Learning rate
        if self.scheduler:
            lrs = [self.optimizer.param_groups[0]['lr'] for _ in range(len(self.train_losses))]
            ax4.plot(lrs)
            ax4.set_title('Learning Rate')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('LR')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_final_report(self, val_s1_targets, val_s1_preds, val_s2_targets, val_s2_preds):
        """Generate final evaluation report"""
        report = {
            'training_summary': {
                'total_epochs': len(self.train_losses),
                'best_val_loss': self.best_val_loss,
                'best_val_acc': self.best_val_acc,
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'final_val_loss': self.val_losses[-1] if self.val_losses else 0
            },
            'stage1_metrics': {
                'accuracy': accuracy_score(val_s1_targets, val_s1_preds),
                'classification_report': classification_report(val_s1_targets, val_s1_preds, 
                                                             target_names=['No Interaction', 'Has Interaction'], 
                                                             output_dict=True)
            }
        }
        
        if len(val_s2_targets) > 0:
            report['stage2_metrics'] = {
                'accuracy': accuracy_score(val_s2_targets, val_s2_preds),
                'classification_report': classification_report(val_s2_targets, val_s2_preds, output_dict=True)
            }
        
        # Save report
        with open(os.path.join(self.save_dir, 'final_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Plot confusion matrices
        self.plot_confusion_matrices(val_s1_targets, val_s1_preds, val_s2_targets, val_s2_preds)
    
    def plot_confusion_matrices(self, val_s1_targets, val_s1_preds, val_s2_targets, val_s2_preds):
        """Plot confusion matrices"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Stage 1 confusion matrix
        cm1 = confusion_matrix(val_s1_targets, val_s1_preds)
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Interaction', 'Has Interaction'],
                    yticklabels=['No Interaction', 'Has Interaction'],
                    ax=axes[0])
        axes[0].set_title('Stage 1: Interaction Detection')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Stage 2 confusion matrix
        if len(val_s2_targets) > 0:
            cm2 = confusion_matrix(val_s2_targets, val_s2_preds)
            labels = ['Walking Together', 'Standing Together', 'Conversation', 'Sitting Together', 'Others']
            sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=axes[1])
            axes[1].set_title('Stage 2: Interaction Type')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')
            plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[1].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Two-Stage Interaction Classifier')
    
    # Dataset
    parser.add_argument('--data_path', type=str, default='D:/1data/imagedata',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Image size (height width)')
    
    # Model
    parser.add_argument('--backbone', type=str, default='mobilenet',
                       choices=['mobilenet'], help='Backbone network')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbone')
    
    # Training
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
    
    # Loss
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
    
    # Logging
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval for training progress')
    
    args = parser.parse_args()
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader, interaction_labels = get_data_loaders(
        args.data_path, args.batch_size, args.num_workers, tuple(args.image_size)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Interaction labels: {interaction_labels}")
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train(train_loader, val_loader)
    
    print(f"Training completed. Results saved to: {trainer.save_dir}")


if __name__ == '__main__':
    main()