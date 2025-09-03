import torch
import torch.nn.functional as F
import argparse
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from two_stage_classifier import TwoStageInteractionClassifier
from dataset import get_data_loaders


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = TwoStageInteractionClassifier(
        backbone_name='mobilenet',  # Default backbone
        pretrained=False,  # Don't need pretrained weights when loading checkpoint
        num_interaction_classes=5
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, test_loader, device, interaction_labels):
    """Evaluate model on test set"""
    model.eval()
    
    all_stage1_preds = []
    all_stage1_targets = []
    all_stage1_probs = []
    
    all_stage2_preds = []
    all_stage2_targets = []
    all_stage2_probs = []
    
    all_final_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            stage1_labels = batch['stage1_label'].to(device)
            stage2_labels = batch['stage2_label'].to(device)
            
            # Get predictions
            predictions = model.predict(images, threshold=0.5)
            
            # Collect stage 1 results
            stage1_preds = (predictions['stage1_probs'][:, 1] > 0.5).long()
            all_stage1_preds.extend(stage1_preds.cpu().numpy())
            all_stage1_targets.extend(stage1_labels.cpu().numpy())
            all_stage1_probs.extend(predictions['stage1_probs'][:, 1].cpu().numpy())
            
            # Collect stage 2 results (only for samples with interactions)
            interaction_mask = stage1_labels == 1
            if interaction_mask.sum() > 0:
                stage2_preds = predictions['interaction_type'][interaction_mask]
                stage2_targets = stage2_labels[interaction_mask]
                stage2_probs = predictions['stage2_probs'][interaction_mask]
                
                all_stage2_preds.extend(stage2_preds.cpu().numpy())
                all_stage2_targets.extend(stage2_targets.cpu().numpy())
                all_stage2_probs.extend(stage2_probs.cpu().numpy())
            
            # Final predictions
            all_final_preds.extend(predictions['final_prediction'].cpu().numpy())
    
    return {
        'stage1': {
            'predictions': np.array(all_stage1_preds),
            'targets': np.array(all_stage1_targets),
            'probabilities': np.array(all_stage1_probs)
        },
        'stage2': {
            'predictions': np.array(all_stage2_preds),
            'targets': np.array(all_stage2_targets),
            'probabilities': np.array(all_stage2_probs) if all_stage2_probs else np.array([])
        },
        'final': {
            'predictions': np.array(all_final_preds)
        }
    }


def print_evaluation_results(results, interaction_labels):
    """Print detailed evaluation results"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Stage 1 Results
    print("\n1. STAGE 1: INTERACTION DETECTION")
    print("-" * 30)
    stage1_acc = accuracy_score(results['stage1']['targets'], results['stage1']['predictions'])
    print(f"Accuracy: {stage1_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(results['stage1']['targets'], results['stage1']['predictions'],
                              target_names=['No Interaction', 'Has Interaction']))
    
    # Stage 2 Results (if any interactions exist)
    if len(results['stage2']['targets']) > 0:
        print("\n2. STAGE 2: INTERACTION TYPE CLASSIFICATION")
        print("-" * 40)
        stage2_acc = accuracy_score(results['stage2']['targets'], results['stage2']['predictions'])
        print(f"Accuracy: {stage2_acc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(results['stage2']['targets'], results['stage2']['predictions'],
                                  target_names=interaction_labels))
        
        # Per-class performance
        print("\nPer-class Performance:")
        for i, label in enumerate(interaction_labels):
            mask = results['stage2']['targets'] == i
            if mask.sum() > 0:
                class_acc = accuracy_score(results['stage2']['targets'][mask], 
                                         results['stage2']['predictions'][mask])
                print(f"  {label}: {class_acc:.4f} ({mask.sum()} samples)")
    
    else:
        print("\n2. STAGE 2: No interaction samples in test set")
    
    # Overall Statistics
    print("\n3. OVERALL STATISTICS")
    print("-" * 20)
    total_samples = len(results['stage1']['targets'])
    interaction_samples = (results['stage1']['targets'] == 1).sum()
    no_interaction_samples = total_samples - interaction_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Interaction samples: {interaction_samples} ({100*interaction_samples/total_samples:.1f}%)")
    print(f"No interaction samples: {no_interaction_samples} ({100*no_interaction_samples/total_samples:.1f}%)")
    
    if len(results['stage2']['targets']) > 0:
        print(f"\nInteraction type distribution:")
        for i, label in enumerate(interaction_labels):
            count = (results['stage2']['targets'] == i).sum()
            print(f"  {label}: {count} ({100*count/len(results['stage2']['targets']):.1f}%)")


def plot_confusion_matrices(results, interaction_labels, save_path):
    """Plot and save confusion matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Stage 1 confusion matrix
    cm1 = confusion_matrix(results['stage1']['targets'], results['stage1']['predictions'])
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Interaction', 'Has Interaction'],
                yticklabels=['No Interaction', 'Has Interaction'],
                ax=axes[0])
    axes[0].set_title('Stage 1: Interaction Detection')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Stage 2 confusion matrix
    if len(results['stage2']['targets']) > 0:
        cm2 = confusion_matrix(results['stage2']['targets'], results['stage2']['predictions'])
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
                    xticklabels=interaction_labels, yticklabels=interaction_labels,
                    ax=axes[1])
        axes[1].set_title('Stage 2: Interaction Type')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        plt.setp(axes[1].get_yticklabels(), rotation=0)
    else:
        axes[1].text(0.5, 0.5, 'No interaction samples\nin test set', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Stage 2: Interaction Type')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm1, cm2 if len(results['stage2']['targets']) > 0 else None


def plot_probability_distributions(results, save_path):
    """Plot probability distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Stage 1 probability distribution
    no_int_probs = results['stage1']['probabilities'][results['stage1']['targets'] == 0]
    has_int_probs = results['stage1']['probabilities'][results['stage1']['targets'] == 1]
    
    axes[0].hist(no_int_probs, bins=30, alpha=0.7, label='No Interaction', density=True)
    axes[0].hist(has_int_probs, bins=30, alpha=0.7, label='Has Interaction', density=True)
    axes[0].set_xlabel('Interaction Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Stage 1: Probability Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Stage 2 confidence distribution
    if len(results['stage2']['probabilities']) > 0:
        max_probs = np.max(results['stage2']['probabilities'], axis=1)
        axes[1].hist(max_probs, bins=30, alpha=0.7, density=True)
        axes[1].set_xlabel('Max Probability (Confidence)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Stage 2: Confidence Distribution')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No interaction samples\nin test set', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Stage 2: Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Two-Stage Interaction Classifier')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='D:/1data/imagedata',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}")
    model, checkpoint = load_model(args.checkpoint_path, device)
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader, interaction_labels = get_data_loaders(
        args.data_path, args.batch_size, args.num_workers
    )
    
    print(f"Test batches: {len(test_loader)}")
    print(f"Interaction labels: {interaction_labels}")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, interaction_labels)
    
    # Print results
    print_evaluation_results(results, interaction_labels)
    
    # Create save directory if specified
    if args.save_results:
        os.makedirs(args.save_results, exist_ok=True)
        
        # Save numerical results
        results_dict = {
            'stage1_accuracy': accuracy_score(results['stage1']['targets'], results['stage1']['predictions']),
            'stage1_report': classification_report(results['stage1']['targets'], results['stage1']['predictions'],
                                                 target_names=['No Interaction', 'Has Interaction'], output_dict=True)
        }
        
        if len(results['stage2']['targets']) > 0:
            results_dict['stage2_accuracy'] = accuracy_score(results['stage2']['targets'], results['stage2']['predictions'])
            results_dict['stage2_report'] = classification_report(results['stage2']['targets'], results['stage2']['predictions'],
                                                                target_names=interaction_labels, output_dict=True)
        
        with open(os.path.join(args.save_results, 'evaluation_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Plot and save visualizations
        cm_path = os.path.join(args.save_results, 'confusion_matrices.png')
        plot_confusion_matrices(results, interaction_labels, cm_path)
        
        prob_path = os.path.join(args.save_results, 'probability_distributions.png')
        plot_probability_distributions(results, prob_path)
        
        print(f"\nResults saved to: {args.save_results}")


if __name__ == '__main__':
    main()