#!/usr/bin/env python3
"""
Test Geometric Stage1 Classifier
Evaluate trained models on test set with detailed analysis
"""

import torch
import torch.nn as nn
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


class GeometricStage1Tester:
    """
    Tester for geometric Stage1 interaction detection
    """

    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(model_path).replace('.pth', '')
        self.results_dir = os.path.join('test_results', f'{model_name}_test_{timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)

        # Load model and checkpoint
        self._load_model_and_checkpoint()

        print(f"GeometricStage1Tester initialized on {self.device}")
        print(f"Results directory: {self.results_dir}")
        print(f"Loaded model from: {model_path}")

    def _load_model_and_checkpoint(self):
        """Load model and checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract config from checkpoint if available
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            # Update current config with checkpoint config for model creation
            for key, value in checkpoint_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    # Add missing attributes from checkpoint
                    setattr(self.config, key, value)

        # Initialize model based on config
        self._initialize_model()

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Extract training info
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'val_accuracy': checkpoint.get('val_accuracy', 'Unknown'),
            'val_f1': checkpoint.get('val_f1', 'Unknown'),
            'val_loss': checkpoint.get('val_loss', 'Unknown')
        }

        print(f"Model trained for {self.checkpoint_info['epoch']} epochs")
        print(f"Best val accuracy: {self.checkpoint_info['val_accuracy']}")
        print(f"Best val F1: {self.checkpoint_info['val_f1']}")

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

    def test_epoch(self, test_loader):
        """Test epoch evaluation"""
        self.model.eval()

        total_loss = 0
        all_predictions = []
        all_targets = []
        all_confidences = []

        criterion = nn.CrossEntropyLoss()

        print("Starting test evaluation...")
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move data to device
                geometric_features = batch['geometric_features'].to(self.device)
                targets = batch['stage1_label'].to(self.device)

                # Forward pass based on model type
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
                loss = criterion(outputs, targets)

                # Get predictions and confidence scores
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]

                # Record metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

                # Progress update
                if batch_idx % 50 == 0:
                    print(f'Test Progress: [{batch_idx}/{len(test_loader)} ({100. * batch_idx / len(test_loader):.0f}%)]')

        test_time = time.time() - start_time
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        print(f'Test completed in {test_time:.1f} seconds')
        print(f'Test Results: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}')

        return avg_loss, accuracy, f1, all_targets, all_predictions, all_confidences

    def generate_detailed_report(self, test_targets, test_predictions, test_confidences,
                               test_loss, test_acc, test_f1):
        """Generate comprehensive test evaluation report"""
        print("\nGenerating detailed test evaluation report...")

        # Classification report
        class_names = ['No Interaction', 'Has Interaction']
        report = classification_report(test_targets, test_predictions, target_names=class_names)

        # Confusion matrix
        cm = confusion_matrix(test_targets, test_predictions)

        # Additional statistics
        test_targets = np.array(test_targets)
        test_predictions = np.array(test_predictions)
        test_confidences = np.array(test_confidences)

        # Per-class metrics
        class_0_mask = test_targets == 0
        class_1_mask = test_targets == 1

        class_0_acc = accuracy_score(test_targets[class_0_mask], test_predictions[class_0_mask])
        class_1_acc = accuracy_score(test_targets[class_1_mask], test_predictions[class_1_mask])

        avg_confidence = np.mean(test_confidences)
        class_0_confidence = np.mean(test_confidences[class_0_mask])
        class_1_confidence = np.mean(test_confidences[class_1_mask])

        # Save comprehensive report
        report_path = os.path.join(self.results_dir, 'detailed_test_report.txt')
        with open(report_path, 'w') as f:
            f.write("Geometric Stage1 Detailed Test Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            # Model and checkpoint info
            f.write("MODEL INFORMATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Training Epochs: {self.checkpoint_info['epoch']}\n")
            f.write(f"Best Val Accuracy: {self.checkpoint_info['val_accuracy']}\n")
            f.write(f"Best Val F1: {self.checkpoint_info['val_f1']}\n\n")

            # Test results
            f.write("TEST RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1 Score: {test_f1:.4f}\n")
            f.write(f"Average Confidence: {avg_confidence:.4f}\n\n")

            # Per-class results
            f.write("PER-CLASS ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"No Interaction - Accuracy: {class_0_acc:.4f}, Avg Confidence: {class_0_confidence:.4f}\n")
            f.write(f"Has Interaction - Accuracy: {class_1_acc:.4f}, Avg Confidence: {class_1_confidence:.4f}\n\n")

            # Classification report
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 30 + "\n")
            f.write(report)
            f.write(f"\n\nCONFUSION MATRIX\n")
            f.write("-" * 30 + "\n")
            f.write(f"True\\Predicted  No Int  Has Int\n")
            f.write(f"No Interaction   {cm[0,0]:6d}  {cm[0,1]:7d}\n")
            f.write(f"Has Interaction  {cm[1,0]:6d}  {cm[1,1]:7d}\n")

            # Model configuration
            f.write(f"\n\nMODEL CONFIGURATION\n")
            f.write("-" * 30 + "\n")
            for key, value in vars(self.config).items():
                f.write(f"{key}: {value}\n")

        print(f"Detailed test report saved to: {report_path}")

        # Save results as JSON for easy parsing
        results_json = {
            'model_info': {
                'type': self.config.model_type,
                'path': self.model_path,
                'training_epochs': self.checkpoint_info['epoch'],
                'val_accuracy': self.checkpoint_info['val_accuracy'],
                'val_f1': self.checkpoint_info['val_f1']
            },
            'test_results': {
                'loss': float(test_loss),
                'accuracy': float(test_acc),
                'f1_score': float(test_f1),
                'avg_confidence': float(avg_confidence)
            },
            'per_class_results': {
                'no_interaction': {
                    'accuracy': float(class_0_acc),
                    'avg_confidence': float(class_0_confidence)
                },
                'has_interaction': {
                    'accuracy': float(class_1_acc),
                    'avg_confidence': float(class_1_confidence)
                }
            },
            'confusion_matrix': cm.tolist(),
            'config': vars(self.config)
        }

        json_path = os.path.join(self.results_dir, 'test_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        return report_path, json_path

    def plot_test_analysis(self, test_targets, test_predictions, test_confidences):
        """Generate test analysis plots"""
        test_targets = np.array(test_targets)
        test_predictions = np.array(test_predictions)
        test_confidences = np.array(test_confidences)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(test_targets, test_predictions)
        class_names = ['No Interaction', 'Has Interaction']

        im = axes[0,0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0,0].set_title('Confusion Matrix')
        tick_marks = np.arange(len(class_names))
        axes[0,0].set_xticks(tick_marks)
        axes[0,0].set_xticklabels(class_names)
        axes[0,0].set_yticks(tick_marks)
        axes[0,0].set_yticklabels(class_names)
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0,0].text(j, i, format(cm[i, j], 'd'),
                              ha="center", va="center",
                              color="white" if cm[i, j] > thresh else "black")

        # 2. Confidence Distribution
        axes[0,1].hist(test_confidences, bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(np.mean(test_confidences), color='red', linestyle='--',
                         label=f'Mean: {np.mean(test_confidences):.3f}')
        axes[0,1].set_title('Confidence Score Distribution')
        axes[0,1].set_xlabel('Confidence Score')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Per-class Confidence
        class_0_conf = test_confidences[test_targets == 0]
        class_1_conf = test_confidences[test_targets == 1]

        axes[1,0].hist([class_0_conf, class_1_conf], bins=30, alpha=0.7,
                      label=['No Interaction', 'Has Interaction'], edgecolor='black')
        axes[1,0].set_title('Confidence by True Class')
        axes[1,0].set_xlabel('Confidence Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 4. Correct vs Incorrect Predictions Confidence
        correct_mask = test_targets == test_predictions
        correct_conf = test_confidences[correct_mask]
        incorrect_conf = test_confidences[~correct_mask]

        axes[1,1].hist([correct_conf, incorrect_conf], bins=30, alpha=0.7,
                      label=[f'Correct ({len(correct_conf)})', f'Incorrect ({len(incorrect_conf)})'],
                      edgecolor='black')
        axes[1,1].set_title('Confidence: Correct vs Incorrect Predictions')
        axes[1,1].set_xlabel('Confidence Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'test_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Test analysis plots saved to: {plot_path}")
        return plot_path


def main():
    parser = argparse.ArgumentParser(description='Test Geometric Stage1 Classifier')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')

    # Data parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')

    # Model parameters (will be overridden by checkpoint if available)
    parser.add_argument('--model_type', type=str, default='adaptive',
                        choices=['adaptive', 'temporal', 'context_aware', 'ensemble'],
                        help='Type of geometric model (will be overridden by checkpoint)')
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

    # Feature options (will be overridden by checkpoint if available)
    parser.add_argument('--use_temporal', action='store_true', default=False,
                        help='Use temporal features (will be overridden by checkpoint)')
    parser.add_argument('--use_scene_context', action='store_true', default=False,
                        help='Use scene context features (will be overridden by checkpoint)')

    # Loading optimization parameters
    parser.add_argument('--loading_strategy', type=str, default='lazy',
                        choices=['cached', 'optimized', 'lazy', 'original'],
                        help='Data loading strategy')

    # Dataset split parameters
    parser.add_argument('--use_custom_splits', action='store_true', default=False,
                        help='Use predefined scene splits instead of percentage-based splits')
    parser.add_argument('--trainset_scenes', type=str, nargs='*', default=None,
                        help='List of scene names for training set (only used with --use_custom_splits)')
    parser.add_argument('--valset_scenes', type=str, nargs='*', default=None,
                        help='List of scene names for validation set (only used with --use_custom_splits)')
    parser.add_argument('--testset_scenes', type=str, nargs='*', default=None,
                        help='List of scene names for test set (only used with --use_custom_splits)')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='Frame sampling interval (1=every frame, 5=every 5th frame)')

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    print(f"Geometric Stage1 Testing Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Data path: {args.data_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Loading strategy: {args.loading_strategy}")

    # Create tester (this will load the model and extract config)
    tester = GeometricStage1Tester(args, args.model_path)

    # Handle custom dataset splits
    if args.use_custom_splits:
        print("\nUsing custom scene splits for test dataset...")
        if args.testset_scenes is None:
            print("Warning: --use_custom_splits specified but --testset_scenes not provided.")
            print("Using default test scene splits")
            from geometric_dataset import create_geometric_data_loaders_with_custom_splits
            _, _, test_loader = create_geometric_data_loaders_with_custom_splits(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                history_length=tester.config.history_length,
                use_temporal=tester.config.use_temporal,
                use_scene_context=tester.config.use_scene_context,
                frame_interval=args.frame_interval
            )
        else:
            print(f"Custom test split: {len(args.testset_scenes)} scenes")
            from geometric_dataset import create_geometric_data_loaders
            _, _, test_loader = create_geometric_data_loaders(
                data_path=args.data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                history_length=tester.config.history_length,
                use_temporal=tester.config.use_temporal,
                use_scene_context=tester.config.use_scene_context,
                trainset_split=args.trainset_scenes or [],
                valset_split=args.valset_scenes or [],
                testset_split=args.testset_scenes,
                use_custom_splits=True,
                frame_interval=args.frame_interval
            )
    else:
        # Create test data loader using the same logic as training
        print(f"\nLoading test data with {args.loading_strategy} strategy...")

        # Choose loading strategy based on the same logic as training
        loading_strategy = args.loading_strategy

    if loading_strategy == 'cached':
        from fast_temporal_cache import create_fast_cached_data_loaders
        _, _, test_loader = create_fast_cached_data_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_temporal=tester.config.use_temporal,
            use_scene_context=tester.config.use_scene_context,
            history_length=tester.config.history_length
        )
    elif loading_strategy == 'optimized':
        from optimized_dataloader import create_optimized_data_loaders
        _, _, test_loader = create_optimized_data_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_temporal=tester.config.use_temporal,
            use_scene_context=tester.config.use_scene_context,
            history_length=tester.config.history_length
        )
    elif loading_strategy == 'lazy':
        from lazy_temporal import create_lazy_data_loaders
        _, _, test_loader = create_lazy_data_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_temporal=tester.config.use_temporal,
            use_scene_context=tester.config.use_scene_context,
            history_length=tester.config.history_length
        )
    else:
        # Fallback to original loader
        _, _, test_loader = create_fast_geometric_data_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            history_length=tester.config.history_length,
            use_temporal=tester.config.use_temporal,
            use_scene_context=tester.config.use_scene_context
        )

    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Test batches: {len(test_loader)}")

    # Run test evaluation
    test_loss, test_acc, test_f1, test_targets, test_predictions, test_confidences = tester.test_epoch(test_loader)

    # Generate detailed reports and plots
    report_path, json_path = tester.generate_detailed_report(
        test_targets, test_predictions, test_confidences,
        test_loss, test_acc, test_f1
    )

    plot_path = tester.plot_test_analysis(test_targets, test_predictions, test_confidences)

    print(f"\nTesting completed!")
    print(f"Results saved in: {tester.results_dir}")
    print(f"  - Detailed report: {report_path}")
    print(f"  - JSON results: {json_path}")
    print(f"  - Analysis plots: {plot_path}")


if __name__ == '__main__':
    main()