#!/usr/bin/env python3
"""
Fast training script for geometric models (without temporal features)
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_temporal_buffer import create_fast_geometric_data_loaders
from train_geometric_stage1 import GeometricStage1Trainer

def main():
    parser = argparse.ArgumentParser(description='Fast Train Geometric Stage1 Classifier')
    
    # Essential parameters
    parser.add_argument('--data_path', type=str, 
                        default=r'C:\assignment\master programme\final\baseline\classificationnet\dataset',
                        help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, default='adaptive',
                        choices=['adaptive', 'context_aware', 'ensemble'],
                        help='Type of geometric model (temporal disabled for speed)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 16],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training control
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    print("Fast Geometric Stage1 Training")
    print("=" * 40)
    print("Note: Temporal features disabled for speed")
    print(f"Model: {args.model_type}")
    print(f"Data path: {args.data_path}")
    
    # Create optimized data loaders (no temporal)
    print("\nCreating fast data loaders...")
    train_loader, val_loader, test_loader = create_fast_geometric_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_temporal=False,  # Disabled for speed
        use_scene_context=True
    )
    
    # Add required attributes for trainer
    args.history_length = 5  # Not used but required
    args.hidden_size = 16    # For context_aware model
    args.num_ensemble_models = 3  # For ensemble model
    args.weight_decay = 1e-4
    args.optimizer = 'adam'
    args.scheduler = 'step'
    args.step_size = 10
    args.weight_regularization = 0.01
    args.sparsity_regularization = 0.01
    args.max_grad_norm = 1.0
    args.log_interval = 100
    args.use_temporal = False
    args.use_scene_context = True
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = GeometricStage1Trainer(args)
    
    # Train model
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)
    
    print("\nFast training completed!")
    print("To enable temporal features later, use the full train_geometric_stage1.py")

if __name__ == '__main__':
    main()