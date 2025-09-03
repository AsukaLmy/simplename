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

from dual_person_classifier import DualPersonInteractionClassifier, DualPersonInteractionLoss
from twostage_training.dual_person_downsampling_dataset import get_dual_person_downsampling_data_loaders, print_dual_person_dataset_statistics
from two_stage_classifier import get_class_weights


class DualPersonStage2DownsamplingTrainer:
    """
    Trainer for Stage 2 (interaction type classification) using dual-person architecture with downsampling
    Loads pretrained stage1 weights and trains only stage2 classifier
    Uses dual-person feature fusion with epoch-based downsampling
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
        
        # Load pretrained stage1 weights if provided
        if config.stage1_checkpoint:
            self.load_stage1_checkpoint(config.stage1_checkpoint)
        
        # Freeze stage1 and backbone - only train stage2
        for param in self.model.backbone_A.parameters():
            param.requires_grad = False
        if not config.shared_backbone:
            for param in self.model.backbone_B.parameters():
                param.requires_grad = False
        for param in self.model.stage1_classifier.parameters():
            param.requires_grad = False
        if hasattr(self.model, 'attention_net'):
            for param in self.model.attention_net.parameters():
                param.requires_grad = False
        
        print("Dual-Person Stage 2 Downsampling Training: Stage1 and backbone frozen")
        print(f"Fusion method: {config.fusion_method}")
        print(f"Shared backbone: {config.shared_backbone}")
        print(f"Training: {config.train_samples_per_epoch} samples per epoch")
        print(f"Validation: {config.val_samples_per_epoch or 'Full dataset'} samples per epoch")
        print(f"Stage1 checkpoint: {config.stage1_checkpoint or 'None'}")
        
        # Create loss function with class weights
        class_weights = get_class_weights() if config.use_class_weights else None
        self.criterion = DualPersonInteractionLoss(
            stage1_weight=0.0,  # Only stage2 loss
            stage2_weight=1.0,
            class_weights=class_weights,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            feature_regularization=False  # No regularization for stage2 only
        ).to(self.device)
        
        # Create optimizer - only for stage2 parameters
        trainable_params = list(self.model.stage2_classifier.parameters())
        
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
        self.epoch_times = []
        self.samples_seen_per_epoch = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        # Create save directory
        self.save_dir = os.path.join(config.save_dir, 
                                   f"dual_person_stage2_downsampling_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        print(f"Experiment directory: {self.save_dir}")
    
    def load_stage1_checkpoint(self, checkpoint_path):
        """Load pretrained stage1 weights"""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Stage1 checkpoint not found: {checkpoint_path}")
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load only compatible weights
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            
            print(f"Loaded Stage1 checkpoint from: {checkpoint_path}")
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
            
        except Exception as e:
            print(f"Warning: Failed to load Stage1 checkpoint: {e}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with downsampling (stage2 only)"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        samples_processed = 0
        stage2_samples_processed = 0
        
        # Update sampler for this epoch (ensures different sampling each epoch)
        if hasattr(train_loader.sampler, 'dataset'):
            train_loader.sampler.dataset.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            person_A_images = batch['person_A_image'].to(self.device)
            person_B_images = batch['person_B_image'].to(self.device)
            stage1_labels = batch['stage1_label'].to(self.device)
            stage2_labels = batch['stage2_label'].to(self.device)
            
            # Forward pass - both stages for loss calculation
            outputs = self.model(person_A_images, person_B_images, stage='both')
            
            # Calculate loss (only stage2)
            loss_dict = self.criterion(outputs, stage1_labels, stage2_labels, stage='stage2')
            loss = loss_dict['total_loss']
            
            # Only process samples with interactions for stage2
            interaction_mask = stage1_labels == 1
            if interaction_mask.sum() > 0:
                stage2_output_filtered = outputs['stage2'][interaction_mask]
                stage2_labels_filtered = stage2_labels[interaction_mask]
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Record loss and predictions
                total_loss += loss.item()
                predictions = torch.argmax(stage2_output_filtered, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(stage2_labels_filtered.cpu().numpy())
                stage2_samples_processed += stage2_labels_filtered.size(0)
            
            samples_processed += person_A_images.size(0)
            
            # Print progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{samples_processed}/{self.config.train_samples_per_epoch} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}, '
                      f'Stage2 samples: {stage2_samples_processed}')
        
        # Calculate epoch metrics
        avg_loss = total_loss / max(len(train_loader), 1)
        accuracy = accuracy_score(all_targets, all_predictions) if len(all_targets) > 0 else 0.0
        f1 = f1_score(all_targets, all_predictions, average='weighted') if len(all_targets) > 0 else 0.0
        
        # Record samples seen this epoch
        self.samples_seen_per_epoch.append(samples_processed)
        
        return avg_loss, accuracy, f1, samples_processed
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch (with optional downsampling)"""
        self.model.eval()
        
        # Update validation sampler for this epoch if using downsampling
        if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'dataset'):
            val_loader.sampler.dataset.set_epoch(epoch)
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        stage2_samples_processed = 0
        
        with torch.no_grad():
            for batch in val_loader:
                person_A_images = batch['person_A_image'].to(self.device)
                person_B_images = batch['person_B_image'].to(self.device)
                stage1_labels = batch['stage1_label'].to(self.device)
                stage2_labels = batch['stage2_label'].to(self.device)
                
                # Forward pass - both stages
                outputs = self.model(person_A_images, person_B_images, stage='both')
                
                # Calculate loss (only stage2)
                loss_dict = self.criterion(outputs, stage1_labels, stage2_labels, stage='stage2')
                total_loss += loss_dict['total_loss'].item()
                
                # Only process samples with interactions for stage2
                interaction_mask = stage1_labels == 1
                if interaction_mask.sum() > 0:
                    stage2_output_filtered = outputs['stage2'][interaction_mask]
                    stage2_labels_filtered = stage2_labels[interaction_mask]
                    
                    # Record predictions for metrics
                    probabilities = torch.softmax(stage2_output_filtered, dim=1)
                    predictions = torch.argmax(stage2_output_filtered, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(stage2_labels_filtered.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    stage2_samples_processed += stage2_labels_filtered.size(0)
        
        # Calculate epoch metrics
        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = accuracy_score(all_targets, all_predictions) if len(all_targets) > 0 else 0.0
        f1 = f1_score(all_targets, all_predictions, average='weighted') if len(all_targets) > 0 else 0.0
        
        return avg_loss, accuracy, f1, all_targets, all_predictions, all_probabilities
    
    def train(self, train_loader, val_loader):
        """Full training loop with downsampling"""
        print(f"Starting Dual-Person Stage 2 downsampling training for {self.config.epochs} epochs...")
        print(f"Original training dataset size: {len(train_loader.dataset)}")
        print(f"Samples per epoch: {self.config.train_samples_per_epoch}")
        print(f"Downsampling ratio: {self.config.train_samples_per_epoch / len(train_loader.dataset):.1%}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        total_start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc, train_f1, samples_processed = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_targets, val_preds, val_probs = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_f1_scores.append(train_f1)
            self.val_f1_scores.append(val_f1)
            
            # Print epoch results
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
            print(f'           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            print(f'           Time: {epoch_time:.2f}s, Samples: {samples_processed}')
            
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
        
        total_time = time.time() - total_start_time
        avg_epoch_time = np.mean(self.epoch_times)
        total_samples = sum(self.samples_seen_per_epoch)
        
        print(f"\nTraining completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
        print(f"Total samples processed: {total_samples}")
        print(f"Average samples per epoch: {total_samples / self.config.epochs:.0f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # Save training statistics
        self.save_training_stats(total_time, avg_epoch_time, total_samples)
    
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
            'config': vars(self.config),
            'training_stats': {
                'samples_per_epoch': self.config.train_samples_per_epoch,
                'fusion_method': self.config.fusion_method,
                'shared_backbone': self.config.shared_backbone,
                'stage1_checkpoint': self.config.stage1_checkpoint
            }
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'{name}.pth'))
        print(f'Checkpoint saved: {name}.pth')
    
    def save_training_stats(self, total_time, avg_epoch_time, total_samples):
        """Save detailed training statistics"""
        stats = {
            'experiment_info': {
                'training_type': 'dual_person_stage2_downsampling',
                'fusion_method': self.config.fusion_method,
                'shared_backbone': self.config.shared_backbone,
                'stage1_checkpoint': self.config.stage1_checkpoint,
                'samples_per_epoch': self.config.train_samples_per_epoch,
                'total_epochs': self.config.epochs,
                'downsampling_enabled': True
            },
            'timing': {
                'total_training_time_seconds': total_time,
                'average_epoch_time_seconds': avg_epoch_time,
                'epoch_times': self.epoch_times
            },
            'sampling': {
                'samples_per_epoch': self.samples_seen_per_epoch,
                'total_samples_processed': total_samples,
                'average_samples_per_epoch': total_samples / self.config.epochs
            },
            'performance': {
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_acc,
                'best_val_f1': self.best_val_f1,
                'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0,
                'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0
            }
        }
        
        with open(os.path.join(self.save_dir, 'training_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def generate_final_report(self, val_targets, val_preds, val_probs):
        """Generate comprehensive evaluation report"""
        print("\nGenerating Dual-Person Stage 2 downsampling evaluation report...")
        
        # Classification report
        interaction_labels = ['walking_together', 'standing_together', 'conversation', 'sitting_together', 'others']
        
        if len(val_targets) > 0:
            report = classification_report(val_targets, val_preds, target_names=interaction_labels)
        else:
            report = "No validation samples with interactions found."
        
        # Save text report
        with open(os.path.join(self.save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Dual-Person Stage 2 Interaction Type Classification Report (Downsampling)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Training Configuration:\n")
            f.write(f"  Fusion method: {self.config.fusion_method}\n")
            f.write(f"  Shared backbone: {self.config.shared_backbone}\n")
            f.write(f"  Stage1 checkpoint: {self.config.stage1_checkpoint or 'None'}\n")
            f.write(f"  Samples per epoch: {self.config.train_samples_per_epoch}\n")
            f.write(f"  Total epochs: {self.config.epochs}\n")
            f.write(f"  Batch size: {self.config.batch_size}\n")
            f.write(f"  Learning rate: {self.config.learning_rate}\n")
            f.write(f"  Use class weights: {self.config.use_class_weights}\n\n")
            f.write("Classification Results:\n")
            f.write(report)
            f.write(f"\n\nBest Results:\n")
            f.write(f"  Best validation loss: {self.best_val_loss:.4f}\n")
            f.write(f"  Best validation accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"  Best validation F1: {self.best_val_f1:.4f}\n")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Plot confusion matrix
        if len(val_targets) > 0:
            self.plot_confusion_matrix(val_targets, val_preds, interaction_labels)
        
        # Plot training efficiency
        self.plot_training_efficiency()
        
        print(f"Evaluation report saved to: {self.save_dir}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_title(f'Stage 2 Loss\n(Fusion: {self.config.fusion_method}, Downsampling: {self.config.train_samples_per_epoch})')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Stage 2 Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 score curves
        axes[1, 0].plot(epochs, self.train_f1_scores, 'b-', label='Training F1')
        axes[1, 0].plot(epochs, self.val_f1_scores, 'r-', label='Validation F1')
        axes[1, 0].set_title('Stage 2 F1 Score')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Epoch time
        axes[1, 1].plot(epochs, self.epoch_times, 'g-')
        axes[1, 1].set_title(f'Training Time per Epoch\n(Avg: {np.mean(self.epoch_times):.2f}s)')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, targets, predictions, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Dual-Person Stage 2 Confusion Matrix\n(Fusion: {self.config.fusion_method}, Downsampling: {self.config.train_samples_per_epoch})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_efficiency(self):
        """Plot training efficiency metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(self.epoch_times) + 1)
        
        # Time per epoch
        axes[0].plot(epochs, self.epoch_times, 'g-o', markersize=3)
        axes[0].set_title('Training Time per Epoch')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].grid(True)
        axes[0].axhline(y=np.mean(self.epoch_times), color='r', linestyle='--', 
                       label=f'Average: {np.mean(self.epoch_times):.2f}s')
        axes[0].legend()
        
        # Samples processed per epoch
        if self.samples_seen_per_epoch:
            axes[1].plot(epochs, self.samples_seen_per_epoch, 'b-o', markersize=3)
            axes[1].set_title('Samples Processed per Epoch')
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Number of Samples')
            axes[1].grid(True)
            axes[1].axhline(y=self.config.train_samples_per_epoch, color='r', linestyle='--', 
                           label=f'Target: {self.config.train_samples_per_epoch}')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Dual-Person Stage 2 Training with Downsampling')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='D:/1data/imagedata',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='Input image size [height, width]')
    
    # Downsampling parameters
    parser.add_argument('--train_samples_per_epoch', type=int, default=10000,
                        help='Number of training samples to use per epoch (downsampling)')
    parser.add_argument('--val_samples_per_epoch', type=int, default=None,
                        help='Number of validation samples to use per epoch (None = no downsampling)')
    parser.add_argument('--test_samples_per_epoch', type=int, default=None,
                        help='Number of test samples to use per epoch (None = no downsampling)')
    parser.add_argument('--balance_train_classes', action='store_true', default=True,
                        help='Maintain 50/50 class balance in downsampled training set')
    
    # Dual-person model parameters
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['mobilenet'], help='Backbone network')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'add', 'subtract', 'multiply', 'attention'],
                        help='Feature fusion method')
    parser.add_argument('--shared_backbone', action='store_true', default=True,
                        help='Share backbone weights between two persons')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                        help='Path to pretrained stage1 model checkpoint')
    
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
    
    # Loss function parameters
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    
    # Dataset parameters
    parser.add_argument('--crop_padding', type=int, default=20,
                        help='Padding around person bounding boxes')
    parser.add_argument('--min_person_size', type=int, default=32,
                        help='Minimum size for person crops')
    
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
    
    print("="*80)
    print("Dual-Person Stage 2 Training with Downsampling")
    print("="*80)
    print(f"Fusion method: {args.fusion_method}")
    print(f"Shared backbone: {args.shared_backbone}")
    print(f"Stage1 checkpoint: {args.stage1_checkpoint or 'None'}")
    print(f"Target samples per epoch: {args.train_samples_per_epoch}")
    print(f"Class balancing: {args.balance_train_classes}")
    print(f"Total epochs: {args.epochs}")
    print("")
    
    # Load data with downsampling
    print("Loading dual-person downsampling dataset...")
    train_loader, val_loader, test_loader, interaction_labels = get_dual_person_downsampling_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_pose=False,  # Not using pose for stage 2
        crop_padding=args.crop_padding,
        min_person_size=args.min_person_size,
        train_samples_per_epoch=args.train_samples_per_epoch,
        balance_train_classes=args.balance_train_classes,
        val_samples_per_epoch=args.val_samples_per_epoch,
        test_samples_per_epoch=args.test_samples_per_epoch
    )
    
    print(f"\nData loaded successfully!")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Print dataset statistics
    print_dual_person_dataset_statistics(train_loader.dataset, "Training Dataset")
    print_dual_person_dataset_statistics(val_loader.dataset, "Validation Dataset")
    
    # Create trainer
    trainer = DualPersonStage2DownsamplingTrainer(args)
    
    # Start training
    print(f"\nStarting training...")
    trainer.train(train_loader, val_loader)
    
    print("\nDual-Person Stage 2 downsampling training completed!")
    print(f"Results saved to: {trainer.save_dir}")
    print("\nTraining Summary:")
    print(f"  Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"  Best validation F1 score: {trainer.best_val_f1:.4f}")
    print(f"  Average time per epoch: {np.mean(trainer.epoch_times):.2f}s")


if __name__ == '__main__':
    main()