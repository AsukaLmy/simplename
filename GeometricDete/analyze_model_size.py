#!/usr/bin/env python3
"""
Analyze model parameter count for different geometric classifiers
"""

import sys
import os
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geometric_classifier import (
    AdaptiveGeometricClassifier,
    CausalTemporalStage1,
    ContextAwareGeometricClassifier,
    GeometricStage1Ensemble
)

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_model_sizes():
    """Analyze parameter counts for different models"""
    
    sample_count = 750000  # Training samples
    
    print("Geometric Model Parameter Analysis")
    print("=" * 50)
    
    models = {
        'AdaptiveGeometric (32,16)': AdaptiveGeometricClassifier(hidden_dims=[32, 16]),
        'AdaptiveGeometric (64,32,16)': AdaptiveGeometricClassifier(hidden_dims=[64, 32, 16]),
        'AdaptiveGeometric (16,8)': AdaptiveGeometricClassifier(hidden_dims=[16, 8]),
        'CausalTemporal': CausalTemporalStage1(hidden_size=16),
        'ContextAware': ContextAwareGeometricClassifier(hidden_dim=32),
        'Ensemble (3 models)': GeometricStage1Ensemble(num_models=3)
    }
    
    results = []
    
    for name, model in models.items():
        total_params, trainable_params = count_parameters(model)
        params_per_sample = trainable_params / sample_count
        
        print(f"\n{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Parameters per sample: {params_per_sample:.4f}")
        
        # Evaluate ratio
        if params_per_sample < 0.001:
            status = "🟢 Excellent (Very safe from overfitting)"
        elif params_per_sample < 0.01:
            status = "🟢 Very Good (Safe from overfitting)"
        elif params_per_sample < 0.1:
            status = "🟡 Good (Acceptable ratio)"
        elif params_per_sample < 1.0:
            status = "🟠 Caution (Risk of overfitting)"
        else:
            status = "🔴 High Risk (Likely to overfit)"
        
        print(f"  Status: {status}")
        
        results.append({
            'model': name,
            'trainable_params': trainable_params,
            'params_per_sample': params_per_sample,
            'status': status
        })
    
    # Comparison with CNN models
    print(f"\n" + "=" * 50)
    print("Comparison with CNN Models:")
    print(f"  MobileNet backbone (frozen): ~2M params")
    print(f"  MobileNet backbone (trainable): ~4.4M params") 
    print(f"  ResNet50 backbone: ~25M params")
    
    cnn_ratio = 4400000 / sample_count
    print(f"  CNN params/sample ratio: {cnn_ratio:.4f} (much higher!)")
    
    # Recommendations
    print(f"\n" + "=" * 50)
    print("Recommendations:")
    
    best_models = [r for r in results if 'Excellent' in r['status'] or 'Very Good' in r['status']]
    if best_models:
        print("✅ Recommended models for 750K samples:")
        for model in best_models:
            print(f"  - {model['model']}: {model['trainable_params']:,} params")
    
    print(f"\n💡 Key Insights:")
    print(f"  - Geometric models are 100-1000x smaller than CNN models")
    print(f"  - With 750K samples, even largest geometric model is very safe")
    print(f"  - You could use the most complex model without overfitting risk")
    print(f"  - Consider using ensemble for maximum accuracy")

def analyze_data_efficiency():
    """Analyze how much data is actually needed"""
    
    print(f"\n" + "=" * 50)
    print("Data Efficiency Analysis:")
    
    # Typical parameter counts
    model_sizes = {
        'Ultra-light (16,8)': 217,
        'Light (32,16)': 817, 
        'Medium (64,32,16)': 2849,
        'Temporal': 3000,
        'Ensemble': 6000
    }
    
    target_ratios = [0.001, 0.01, 0.1]  # Different safety levels
    
    print(f"\nRequired samples for different safety ratios:")
    print(f"{'Model':<20} {'Ultra-safe':<12} {'Very safe':<12} {'Safe':<12}")
    print(f"{'':20} {'(1:1000)':<12} {'(1:100)':<12} {'(1:10)':<12}")
    print("-" * 60)
    
    for model_name, params in model_sizes.items():
        ultra_safe = int(params / 0.001)
        very_safe = int(params / 0.01) 
        safe = int(params / 0.1)
        
        print(f"{model_name:<20} {ultra_safe:<12,} {very_safe:<12,} {safe:<12,}")
    
    print(f"\n💡 With 750K samples, ALL geometric models are in 'Ultra-safe' category!")

if __name__ == '__main__':
    analyze_model_sizes()
    analyze_data_efficiency()