import torch
import torch.nn as nn
import torch.nn.functional as F
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

from two_stage_classifier import TwoStageInteractionClassifier, InteractionLoss, get_class_weights
from twostage_training.optimized_dataset import get_optimized_data_loaders


class Stage2Trainer:
    """
    Trainer specifically for Stage 2 (interaction type classification)
    Loads pretrained Stage 1 model and only trains the stage2 classifier
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = TwoStageInteractionClassifier(
            backbone_name=config.backbone,
            pretrained=False,  # We'll load from stage1 checkpoint
            num_interaction_classes=5
        ).to(self.device)
        
        # Load Stage 1 pretrained weights
        if config.stage1_checkpoint:
            self.load_stage1_checkpoint(config.stage1_checkpoint)
        else:
            print("Warning: No Stage 1 checkpoint provided. Training from scratch.")
        
        # Freeze backbone and stage1 classifier - only train stage2
        if config.freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Stage 2 Training: Backbone frozen")
        
        for param in self.model.stage1_classifier.parameters():
            param.requires_grad = False
        print("Stage 2 Training: Stage1 classifier frozen")
        
        # Create loss function for Stage 2 with class weights and focal loss
        class_weights = get_class_weights() if config.use_class_weights else None
        self.criterion = InteractionLoss(
            stage1_weight=0.0,  # No stage1 loss
            stage2_weight=1.0,  # Only stage2 loss
            class_weights=class_weights,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        ).to(self.device)
        
        # Create optimizer - only for stage2 classifier (and optionally backbone)
        trainable_params = []
        if not config.freeze_backbone:
            trainable_params.extend(self.model.backbone.parameters())
        trainable_params.extend(self.model.stage2_classifier.parameters())
        
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
                                   f"stage2_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
    
    def load_stage1_checkpoint(self, checkpoint_path):
        """Load Stage 1 pretrained weights"""
        print(f"Loading Stage 1 checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load the weights
            self.model.load_state_dict(state_dict, strict=False)
            print("Stage 1 checkpoint loaded successfully")
            
            # Print some info about the loaded model
            if 'epoch' in checkpoint:
                print(f"Loaded from epoch: {checkpoint['epoch']}")
            if 'val_accuracy' in checkpoint:
                print(f"Stage 1 validation accuracy: {checkpoint['val_accuracy']:.4f}")
                
        except Exception as e:
            print(f"Error loading Stage 1 checkpoint: {e}")
            print("Continuing with random initialization...")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        processed_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            stage1_labels = batch['stage1_label'].to(self.device)
            stage2_labels = batch['stage2_label'].to(self.device)
            
            # Only process samples with interactions (stage1_labels == 1)
            interaction_mask = stage1_labels == 1
            
            if interaction_mask.sum() == 0:
                continue  # Skip batch if no interactions
            
            # Filter batch to only interaction samples
            images_filtered = images[interaction_mask]
            stage1_labels_filtered = stage1_labels[interaction_mask]
            stage2_labels_filtered = stage2_labels[interaction_mask]
            
            # Forward pass - both stages (need stage1 for loss calculation)
            outputs = self.model(images_filtered, stage='both')
            
            # Calculate loss - only stage2
            loss_dict = self.criterion(outputs, stage1_labels_filtered, stage2_labels_filtered, stage='stage2')
            loss = loss_dict['total_loss']
            
            # Skip if loss is 0 (no valid stage2 samples)
            if loss.item() == 0:
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            total_loss += loss.item()
            processed_samples += 1
            
            # Record predictions for metrics
            stage2_output = outputs['stage2']
            predictions = torch.argmax(stage2_output, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(stage2_labels_filtered.cpu().numpy())
            
            # Print progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{processed_samples} interaction batches]\tLoss: {loss.item():.6f}')
        
        if processed_samples == 0:
            print(f"Warning: No interaction samples found in epoch {epoch}")
            return 0, 0, 0
        
        # Calculate epoch metrics
        avg_loss = total_loss / processed_samples
        accuracy = accuracy_score(all_targets, all_predictions) if len(all_targets) > 0 else 0
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0) if len(all_targets) > 0 else 0
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        processed_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                stage1_labels = batch['stage1_label'].to(self.device)
                stage2_labels = batch['stage2_label'].to(self.device)
                
                # Only process samples with interactions
                interaction_mask = stage1_labels == 1
                
                if interaction_mask.sum() == 0:
                    continue
                
                # Filter batch to only interaction samples
                images_filtered = images[interaction_mask]
                stage1_labels_filtered = stage1_labels[interaction_mask]
                stage2_labels_filtered = stage2_labels[interaction_mask]
                
                # Forward pass
                outputs = self.model(images_filtered, stage='both')
                
                # Calculate loss
                loss_dict = self.criterion(outputs, stage1_labels_filtered, stage2_labels_filtered, stage='stage2')
                loss = loss_dict['total_loss']
                
                if loss.item() == 0:
                    continue
                
                total_loss += loss.item()
                processed_samples += 1
                
                # Record predictions for metrics
                stage2_output = outputs['stage2']
                probabilities = torch.softmax(stage2_output, dim=1)
                predictions = torch.argmax(stage2_output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(stage2_labels_filtered.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        if processed_samples == 0:
            return 0, 0, 0, [], [], []
        
        # Calculate epoch metrics
        avg_loss = total_loss / processed_samples
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return avg_loss, accuracy, f1, all_targets, all_predictions, all_probabilities
    
    def train(self, train_loader, val_loader, interaction_labels):
        """Full training loop"""
        print(f"Starting Stage 2 training for {self.config.epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Interaction labels: {interaction_labels}")
        
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
        if len(val_targets) > 0:
            self.generate_final_report(val_targets, val_preds, val_probs, interaction_labels)
        
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
    
    def generate_final_report(self, val_targets, val_preds, val_probs, interaction_labels):
        """Generate comprehensive evaluation report"""
        print("\nGenerating Stage 2 evaluation report...")
        
        # Classification report
        if len(interaction_labels) >= max(max(val_targets), max(val_preds)) + 1:
            class_names = interaction_labels
        else:
            class_names = [f'Class_{i}' for i in range(max(max(val_targets), max(val_preds)) + 1)]
        
        report = classification_report(val_targets, val_preds, target_names=class_names, zero_division=0)
        
        # Save text report
        with open(os.path.join(self.save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Stage 2 Interaction Type Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
            f.write(f"\n\nBest validation loss: {self.best_val_loss:.4f}\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"Best validation F1: {self.best_val_f1:.4f}\n")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(val_targets, val_preds, class_names)
        
        # Plot class distribution
        self.plot_class_distribution(val_targets, class_names)
        
        print(f"Evaluation report saved to: {self.save_dir}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_title('Stage 2 Training and Validation Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Stage 2 Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 score curves
        axes[1, 0].plot(epochs, self.train_f1_scores, 'b-', label='Training F1')
        axes[1, 0].plot(epochs, self.val_f1_scores, 'r-', label='Validation F1')
        axes[1, 0].set_title('Stage 2 Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if self.scheduler:
            try:
                lrs = [self.scheduler.get_last_lr()[0] for _ in epochs]
                axes[1, 1].plot(epochs, lrs, 'g-')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_xlabel('Epochs')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True)
            except:
                axes[1, 1].text(0.5, 0.5, 'Learning rate not available', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Learning Rate Schedule')
        else:
            axes[1, 1].text(0.5, 0.5, 'No scheduler used', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, targets, predictions, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Stage 2 Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_distribution(self, targets, class_names):
        """Plot class distribution in validation set"""
        from collections import Counter
        class_counts = Counter(targets)
        
        plt.figure(figsize=(12, 6))
        classes = [class_names[i] for i in sorted(class_counts.keys())]
        counts = [class_counts[i] for i in sorted(class_counts.keys())]
        
        plt.bar(classes, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(classes)])
        plt.title('Stage 2 Validation Set Class Distribution')
        plt.xlabel('Interaction Type')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts) * 0.01, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 2 Training: Interaction Type Classification')
    
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
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Path to Stage 1 pretrained checkpoint')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze backbone weights during Stage 2 training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate (lower than Stage 1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'none'], help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=15,
                        help='Step size for StepLR scheduler')
    
    # Loss parameters
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    
    # Dataset optimization parameters
    parser.add_argument('--distance_multiplier', type=float, default=3.0,
                        help='Distance multiplier for pairing constraint')
    parser.add_argument('--max_negatives_per_frame', type=int, default=5,
                        help='Maximum negative samples per frame')
    parser.add_argument('--stage2_balance_strategy', type=str, default='oversample',
                        choices=['oversample', 'undersample', 'none'],
                        help='Stage 2 sample balancing strategy')
    parser.add_argument('--use_group_sampling', action='store_true', default=True,
                        help='Use group-based positive sampling to reduce redundancy')
    parser.add_argument('--max_group_samples_ratio', type=float, default=1.0,
                        help='Maximum samples per group as ratio of group size')
    
    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Checkpoint saving interval')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if Stage 1 checkpoint exists
    if not os.path.exists(args.stage1_checkpoint):
        print(f"Error: Stage 1 checkpoint not found: {args.stage1_checkpoint}")
        print("Please train Stage 1 first using train_stage1.py")
        return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data with Stage 2 balancing
    print("Loading optimized dataset for Stage 2...")
    train_loader, val_loader, test_loader, interaction_labels = get_optimized_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_pose=False,  # Not using pose for now
        distance_multiplier=args.distance_multiplier,
        max_negatives_per_frame=args.max_negatives_per_frame,
        stage1_balance_ratio=1.0,  # Not important for Stage 2 training
        stage2_balance_strategy=args.stage2_balance_strategy,
        use_group_sampling=args.use_group_sampling,
        max_group_samples_ratio=args.max_group_samples_ratio
    )
    
    print(f"Data loaded successfully!")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = Stage2Trainer(args)
    
    # Start training
    trainer.train(train_loader, val_loader, interaction_labels)
    
    print("Stage 2 training completed!")
    print(f"Results saved to: {trainer.save_dir}")


if __name__ == '__main__':
    main()