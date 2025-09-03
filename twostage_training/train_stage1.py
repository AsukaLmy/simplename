import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import argparse
import os
import sys
import time
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from two_stage_classifier import TwoStageInteractionClassifier
from twostage_training.optimized_dataset import get_optimized_data_loaders


class Stage1Trainer:
    """
    Trainer specifically for Stage 1 (binary interaction detection)
    Only trains the stage1 classifier while keeping stage2 frozen
    """
    
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
        
        # Freeze stage2 classifier - only train stage1
        for param in self.model.stage2_classifier.parameters():
            param.requires_grad = False
        
        print("Stage 1 Training: Stage2 classifier frozen")
        
        # Create loss function for binary classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Create optimizer - only for stage1 and backbone parameters
        trainable_params = []
        trainable_params.extend(self.model.backbone.parameters())
        trainable_params.extend(self.model.stage1_classifier.parameters())
        
        if config.optimizer == 'adam':
            self.optimizer = optim.Adam(trainable_params, 
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        elif config.optimizer == 'sgd':
            self.optimizer = optim.SGD(trainable_params,
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
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        # Create save directory
        self.save_dir = os.path.join(config.save_dir, 
                                   f"stage1_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            stage1_labels = batch['stage1_label'].to(self.device)
            
            # Forward pass - only stage 1
            outputs = self.model(images, stage='stage1')
            stage1_output = outputs['stage1']
            
            # Calculate loss
            loss = self.criterion(stage1_output, stage1_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            total_loss += loss.item()
            
            # Record predictions for metrics
            predictions = torch.argmax(stage1_output, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(stage1_labels.cpu().numpy())
            
            # Print progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                stage1_labels = batch['stage1_label'].to(self.device)
                
                # Forward pass - only stage 1
                outputs = self.model(images, stage='stage1')
                stage1_output = outputs['stage1']
                
                # Calculate loss
                loss = self.criterion(stage1_output, stage1_labels)
                total_loss += loss.item()
                
                # Record predictions for metrics
                probabilities = torch.softmax(stage1_output, dim=1)
                predictions = torch.argmax(stage1_output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(stage1_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1, all_targets, all_predictions, all_probabilities
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"Starting Stage 1 training for {self.config.epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_targets, val_preds, val_probs = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_f1_scores.append(train_f1)
            self.val_f1_scores.append(val_f1)
            
            # Print epoch results
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
            print(f'           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            
            # Save best models
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_loss', epoch, val_loss, val_acc, val_f1)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_accuracy', epoch, val_loss, val_acc, val_f1)
            
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_checkpoint('best_f1', epoch, val_loss, val_acc, val_f1)
            
            # Save regular checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, val_loss, val_acc, val_f1)
        
        # Save final model
        self.save_checkpoint('final', self.config.epochs, val_loss, val_acc, val_f1)
        
        # Generate final evaluation report
        self.generate_final_report(val_targets, val_preds, val_probs)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
    
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
            'config': vars(self.config)
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'{name}.pth'))
        print(f'Checkpoint saved: {name}.pth')
    
    def generate_final_report(self, val_targets, val_preds, val_probs):
        """Generate comprehensive evaluation report"""
        print("\nGenerating Stage 1 evaluation report...")
        
        # Classification report
        class_names = ['No Interaction', 'Has Interaction']
        report = classification_report(val_targets, val_preds, target_names=class_names)
        
        # Save text report
        with open(os.path.join(self.save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Stage 1 Binary Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write(f"\n\nBest validation loss: {self.best_val_loss:.4f}\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"Best validation F1: {self.best_val_f1:.4f}\n")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(val_targets, val_preds, class_names)
        
        # Plot probability distributions
        self.plot_probability_distributions(val_targets, val_probs)
        
        print(f"Evaluation report saved to: {self.save_dir}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_title('Stage 1 Training and Validation Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Stage 1 Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 score curves
        axes[1, 0].plot(epochs, self.train_f1_scores, 'b-', label='Training F1')
        axes[1, 0].plot(epochs, self.val_f1_scores, 'r-', label='Validation F1')
        axes[1, 0].set_title('Stage 1 Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if self.scheduler:
            lrs = [self.scheduler.get_last_lr()[0] for _ in epochs]
            axes[1, 1].plot(epochs, lrs, 'g-')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No scheduler used', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, targets, predictions, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Stage 1 Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_probability_distributions(self, targets, probabilities):
        """Plot probability distributions for each class"""
        probabilities = np.array(probabilities)
        targets = np.array(targets)
        
        plt.figure(figsize=(12, 5))
        
        # Plot for "No Interaction" class
        plt.subplot(1, 2, 1)
        no_interaction_probs = probabilities[targets == 0, 0]  # P(No Interaction)
        has_interaction_probs_wrong = probabilities[targets == 1, 0]  # P(No Interaction) when true label is "Has Interaction"
        
        plt.hist(no_interaction_probs, bins=50, alpha=0.7, label='True No Interaction', color='blue')
        plt.hist(has_interaction_probs_wrong, bins=50, alpha=0.7, label='True Has Interaction', color='red')
        plt.title('P(No Interaction) Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot for "Has Interaction" class
        plt.subplot(1, 2, 2)
        has_interaction_probs = probabilities[targets == 1, 1]  # P(Has Interaction)
        no_interaction_probs_wrong = probabilities[targets == 0, 1]  # P(Has Interaction) when true label is "No Interaction"
        
        plt.hist(has_interaction_probs, bins=50, alpha=0.7, label='True Has Interaction', color='red')
        plt.hist(no_interaction_probs_wrong, bins=50, alpha=0.7, label='True No Interaction', color='blue')
        plt.title('P(Has Interaction) Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 1 Training: Binary Interaction Detection')
    
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
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'none'], help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for StepLR scheduler')
    
    # Dataset optimization parameters
    parser.add_argument('--distance_multiplier', type=float, default=3.0,
                        help='Distance multiplier for pairing constraint')
    parser.add_argument('--max_negatives_per_frame', type=int, default=5,
                        help='Maximum negative samples per frame')
    parser.add_argument('--stage1_balance_ratio', type=float, default=1.0,
                        help='Negative to positive ratio for stage 1')
    parser.add_argument('--use_group_sampling', action='store_true', default=True,
                        help='Use group-based positive sampling to reduce redundancy')
    parser.add_argument('--max_group_samples_ratio', type=float, default=1.0,
                        help='Maximum samples per group as ratio of group size')
    
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
    
    # Load data
    print("Loading optimized dataset...")
    train_loader, val_loader, test_loader, interaction_labels = get_optimized_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_pose=False,  # Not using pose for stage 1
        distance_multiplier=args.distance_multiplier,
        max_negatives_per_frame=args.max_negatives_per_frame,
        stage1_balance_ratio=args.stage1_balance_ratio,
        stage2_balance_strategy='none',  # No stage 2 balancing needed for stage 1 training
        use_group_sampling=args.use_group_sampling,
        max_group_samples_ratio=args.max_group_samples_ratio
    )
    
    print(f"Data loaded successfully!")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = Stage1Trainer(args)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("Stage 1 training completed!")
    print(f"Results saved to: {trainer.save_dir}")


if __name__ == '__main__':
    main()