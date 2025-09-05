import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from optimized_temporal_buffer import create_fast_geometric_data_loaders
from geometric_classifier import (
    AdaptiveGeometricClassifier, 
    CausalTemporalStage1, 
    ContextAwareGeometricClassifier,
    GeometricStage1Ensemble,
    compute_adaptive_loss
)


class GeometricStage1Trainer:
    """
    Trainer for geometric Stage1 interaction detection
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join('checkpoints', f'geometric_stage1_{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
        
        # Setup training components
        self._setup_training()
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        # Save configuration
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)
        
        print(f"GeometricStage1Trainer initialized on {self.device}")
        print(f"Save directory: {self.save_dir}")
    
    def _initialize_model(self):
        """Initialize the geometric model based on config"""
        if self.config.model_type == 'adaptive':
            self.model = AdaptiveGeometricClassifier(
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout
            )
        elif self.config.model_type == 'temporal':
            self.model = CausalTemporalStage1(
                history_length=self.config.history_length,
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout
            )
        elif self.config.model_type == 'context_aware':
            self.model = ContextAwareGeometricClassifier(
                hidden_dim=self.config.hidden_size
            )
        elif self.config.model_type == 'ensemble':
            self.model = GeometricStage1Ensemble(
                num_models=self.config.num_ensemble_models
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {self.config.model_type}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_training(self):
        """Setup optimizer, criterion, scheduler"""
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
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
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
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
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            geometric_features = batch['geometric_features'].to(self.device)
            targets = batch['stage1_label'].to(self.device)
            
            # Prepare model inputs based on model type
            if self.config.model_type == 'temporal':
                history_geometric = batch['history_geometric'].to(self.device)
                motion_features = batch['motion_features'].to(self.device)
                scene_context = batch['scene_context'].to(self.device)
                
                outputs = self.model(
                    geometric_features, history_geometric, 
                    motion_features, scene_context
                )
            elif self.config.model_type == 'context_aware' or self.config.model_type == 'ensemble':
                scene_context = batch['scene_context'].to(self.device)
                outputs = self.model(geometric_features, scene_context)
            else:
                outputs = self.model(geometric_features)
            
            # Compute loss
            if self.config.model_type == 'adaptive' and hasattr(self.model, 'feature_weights'):
                loss = compute_adaptive_loss(
                    outputs, targets, self.model.feature_weights,
                    self.config.weight_regularization, self.config.sparsity_regularization
                )
            else:
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            # Record metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Print progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch {epoch}: [{batch_idx * len(geometric_features)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                geometric_features = batch['geometric_features'].to(self.device)
                targets = batch['stage1_label'].to(self.device)
                
                # Forward pass
                if self.config.model_type == 'temporal':
                    history_geometric = batch['history_geometric'].to(self.device)
                    motion_features = batch['motion_features'].to(self.device)
                    scene_context = batch['scene_context'].to(self.device)
                    
                    outputs = self.model(
                        geometric_features, history_geometric,
                        motion_features, scene_context
                    )
                elif self.config.model_type == 'context_aware' or self.config.model_type == 'ensemble':
                    scene_context = batch['scene_context'].to(self.device)
                    outputs = self.model(geometric_features, scene_context)
                else:
                    outputs = self.model(geometric_features)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Record metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1, all_targets, all_predictions
    
    def test_epoch(self, test_loader):
        """Test epoch evaluation (same logic as validation)"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                geometric_features = batch['geometric_features'].to(self.device)
                targets = batch['stage1_label'].to(self.device)
                
                # Forward pass
                if self.config.model_type == 'temporal':
                    history_geometric = batch['history_geometric'].to(self.device)
                    motion_features = batch['motion_features'].to(self.device)
                    scene_context = batch['scene_context'].to(self.device)
                    
                    outputs = self.model(
                        geometric_features, history_geometric,
                        motion_features, scene_context
                    )
                elif self.config.model_type == 'context_aware' or self.config.model_type == 'ensemble':
                    scene_context = batch['scene_context'].to(self.device)
                    outputs = self.model(geometric_features, scene_context)
                else:
                    outputs = self.model(geometric_features)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Record metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
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
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
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
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}', epoch, val_loss, val_acc, val_f1)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.1f} seconds')
        print(f'Best validation accuracy: {self.best_val_acc:.4f}')
        print(f'Best validation F1: {self.best_val_f1:.4f}')
        
        # Generate final report
        self.generate_final_report(val_targets, val_predictions)
        self.plot_training_curves()
        
        # Analyze feature importance if available
        self.analyze_feature_importance()
        
        # Test evaluation if test_loader is provided
        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_loss, test_acc, test_f1, test_targets, test_predictions = self.test_epoch(test_loader)
            
            print(f'Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}')
            
            # Generate test report
            self.generate_test_report(test_targets, test_predictions, test_loss, test_acc, test_f1)
    
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
    
    def generate_final_report(self, val_targets, val_predictions):
        """Generate evaluation report"""
        print("\nGenerating Geometric Stage1 evaluation report...")
        
        # Classification report
        class_names = ['No Interaction', 'Has Interaction']
        report = classification_report(val_targets, val_predictions, target_names=class_names)
        
        # Confusion matrix
        cm = confusion_matrix(val_targets, val_predictions)
        
        # Save report
        with open(os.path.join(self.save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Geometric Stage1 Binary Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"Best Validation F1: {self.best_val_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write(f"\n\nConfusion Matrix:\n{cm}\n")
        
        print(f"Evaluation report saved to {self.save_dir}")
    
    def generate_test_report(self, test_targets, test_predictions, test_loss, test_acc, test_f1):
        """Generate test evaluation report"""
        print("\nGenerating test evaluation report...")
        
        # Classification report
        class_names = ['No Interaction', 'Has Interaction']
        report = classification_report(test_targets, test_predictions, target_names=class_names)
        
        # Confusion matrix
        cm = confusion_matrix(test_targets, test_predictions)
        
        # Save test report
        with open(os.path.join(self.save_dir, 'test_evaluation_report.txt'), 'w') as f:
            f.write("Geometric Stage1 Test Set Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1 Score: {test_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))
        
        print(f"Test evaluation report saved to: {os.path.join(self.save_dir, 'test_evaluation_report.txt')}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()
    
    def analyze_feature_importance(self):
        """Analyze and save feature importance if model supports it"""
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            
            print("\nLearned Feature Importance:")
            for feature, weight in importance:
                print(f"  {feature}: {weight:.4f}")
            
            # Save to file
            with open(os.path.join(self.save_dir, 'feature_importance.json'), 'w') as f:
                json.dump(dict(importance), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train Geometric Stage1 Classifier')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='adaptive',
                        choices=['adaptive', 'temporal', 'context_aware', 'ensemble'],
                        help='Type of geometric model')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 16],
                        help='Hidden layer dimensions for adaptive model')
    parser.add_argument('--hidden_size', type=int, default=16,
                        help='Hidden size for temporal/context models')
    parser.add_argument('--history_length', type=int, default=5,
                        help='Length of temporal history')
    parser.add_argument('--num_ensemble_models', type=int, default=3,
                        help='Number of models in ensemble')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
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
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for step scheduler')
    
    # Regularization parameters
    parser.add_argument('--weight_regularization', type=float, default=0.01,
                        help='Feature weight regularization')
    parser.add_argument('--sparsity_regularization', type=float, default=0.01,
                        help='Feature sparsity regularization')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Training control
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    
    # Feature options
    parser.add_argument('--use_temporal', action='store_true', default=True,
                        help='Use temporal features')
    parser.add_argument('--use_scene_context', action='store_true', default=True,
                        help='Use scene context features')
    
    args = parser.parse_args()
    
    # Create data loaders
    print("Loading geometric data...")
    train_loader, val_loader, test_loader = create_fast_geometric_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history_length=args.history_length,
        use_temporal=args.use_temporal,
        use_scene_context=args.use_scene_context
    )
    
    # Create trainer
    trainer = GeometricStage1Trainer(args)
    
    # Train model
    trainer.train(train_loader, val_loader, test_loader)
    
    print("Training completed!")


if __name__ == '__main__':
    main()