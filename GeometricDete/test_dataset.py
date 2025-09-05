#!/usr/bin/env python3
"""
Test script for JRDB geometric dataset loading
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geometric_dataset import GeometricDualPersonDataset, create_geometric_data_loaders


def test_dataset_loading():
    """Test dataset loading with JRDB format"""
    
    # Use your actual dataset path
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    print("Testing JRDB Geometric Dataset Loading...")
    print(f"Data path: {data_path}")
    
    try:
        # Test basic dataset creation
        print("\n=== Testing Train Dataset ===")
        train_dataset = GeometricDualPersonDataset(
            data_path=data_path,
            split='train',
            history_length=3,
            use_temporal=False,  # Disable temporal for initial test
            use_scene_context=True
        )
        
        print(f"Train dataset loaded: {len(train_dataset)} samples")
        
        if len(train_dataset) > 0:
            # Test first sample
            sample = train_dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Geometric features shape: {sample['geometric_features'].shape}")
            print(f"Stage1 label: {sample['stage1_label'].item()}")
            print(f"Frame ID: {sample['frame_id']}")
            print(f"Scene context: {sample['scene_context']}")
            
            # Test class distribution
            class_dist = train_dataset.get_class_distribution()
            print(f"Class distribution: {class_dist}")
        
        # Test validation dataset
        print("\n=== Testing Val Dataset ===")
        val_dataset = GeometricDualPersonDataset(
            data_path=data_path,
            split='val',
            history_length=3,
            use_temporal=False,
            use_scene_context=True
        )
        
        print(f"Val dataset loaded: {len(val_dataset)} samples")
        
        if len(val_dataset) > 0:
            val_dist = val_dataset.get_class_distribution()
            print(f"Val class distribution: {val_dist}")
        
        print("\n=== Dataset Loading Successful! ===")
        return True
        
    except Exception as e:
        print(f"Error during dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loaders():
    """Test data loader creation"""
    
    data_path = r'C:\assignment\master programme\final\baseline\classificationnet\dataset'
    
    print("\n=== Testing Data Loaders ===")
    
    try:
        train_loader, val_loader, test_loader = create_geometric_data_loaders(
            data_path=data_path,
            batch_size=4,  # Small batch for testing
            num_workers=0,  # Avoid multiprocessing issues
            history_length=3,
            use_temporal=False,  # Disable temporal for initial test
            use_scene_context=True
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches") 
        print(f"Test loader: {len(test_loader)} batches")
        
        # Test one batch
        if len(train_loader) > 0:
            print("\n=== Testing First Batch ===")
            for batch_idx, batch in enumerate(train_loader):
                print(f"Batch {batch_idx}:")
                print(f"  Geometric features: {batch['geometric_features'].shape}")
                print(f"  Stage1 labels: {batch['stage1_label'].shape}")
                print(f"  Scene context: {batch['scene_context'].shape}")
                
                # Check for NaN or infinity
                import torch
                geo_features = batch['geometric_features']
                if torch.isnan(geo_features).any():
                    print("  WARNING: NaN values found in geometric features!")
                if torch.isinf(geo_features).any():
                    print("  WARNING: Inf values found in geometric features!")
                
                print(f"  Sample geometric features:\n{geo_features[0]}")
                break
        
        print("=== Data Loaders Test Successful! ===")
        return True
        
    except Exception as e:
        print(f"Error during data loader testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("JRDB Geometric Dataset Test Suite")
    print("=" * 50)
    
    # Test dataset loading
    dataset_ok = test_dataset_loading()
    
    if dataset_ok:
        # Test data loaders
        loader_ok = test_data_loaders()
        
        if loader_ok:
            print("\n" + "=" * 50)
            print("✅ All tests passed! Dataset is ready for training.")
        else:
            print("\n" + "=" * 50)  
            print("❌ Data loader test failed!")
    else:
        print("\n" + "=" * 50)
        print("❌ Dataset loading test failed!")


if __name__ == '__main__':
    main()