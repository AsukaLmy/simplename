import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
from collections import defaultdict, Counter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_person_dataset import DualPersonJRDBInteractionDataset
from torchvision import transforms


class EpochDownsampleSampler(Sampler):
    """
    Sampler that randomly samples a fixed number of samples per epoch while maintaining class balance
    Compatible with dual-person architecture
    """
    
    def __init__(self, dataset, samples_per_epoch=10000, balance_classes=True):
        """
        Args:
            dataset: The dataset to sample from
            samples_per_epoch: Number of samples to use per epoch
            balance_classes: Whether to maintain class balance (50/50 pos/neg for stage1)
        """
        self.dataset = dataset
        self.samples_per_epoch = min(samples_per_epoch, len(dataset))
        self.balance_classes = balance_classes
        
        # Analyze dataset to find class indices
        self.positive_indices = []
        self.negative_indices = []
        
        for idx, sample in enumerate(dataset.samples):
            if sample['has_interaction'] == 1:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)
        
        print(f"Dual-person dataset has {len(self.positive_indices)} positive and {len(self.negative_indices)} negative samples")
        print(f"Downsampling to {self.samples_per_epoch} samples per epoch")
    
    def __iter__(self):
        """Generate indices for this epoch"""
        if self.balance_classes:
            # Balanced sampling: 50/50 pos/neg
            samples_per_class = self.samples_per_epoch // 2
            
            # Sample positive indices
            if len(self.positive_indices) >= samples_per_class:
                pos_sampled = random.sample(self.positive_indices, samples_per_class)
            else:
                # If not enough positive samples, sample with replacement
                pos_sampled = random.choices(self.positive_indices, k=samples_per_class)
            
            # Sample negative indices
            if len(self.negative_indices) >= samples_per_class:
                neg_sampled = random.sample(self.negative_indices, samples_per_class)
            else:
                # If not enough negative samples, sample with replacement
                neg_sampled = random.choices(self.negative_indices, k=samples_per_class)
            
            # Combine and shuffle
            epoch_indices = pos_sampled + neg_sampled
            random.shuffle(epoch_indices)
            
        else:
            # Random sampling without class balancing
            all_indices = list(range(len(self.dataset)))
            epoch_indices = random.sample(all_indices, self.samples_per_epoch)
        
        return iter(epoch_indices)
    
    def __len__(self):
        return self.samples_per_epoch


class DownsamplingDualPersonJRDBDataset(DualPersonJRDBInteractionDataset):
    """
    Extension of DualPersonJRDBInteractionDataset with downsampling support
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_number = 0  # Track current epoch for logging
    
    def set_epoch(self, epoch):
        """Set current epoch number (for logging purposes)"""
        self.epoch_number = epoch


def get_dual_person_downsampling_data_loaders(data_path, batch_size=16, num_workers=4, image_size=(224, 224), 
                                              use_pose=False, crop_padding=20, min_person_size=32,
                                              train_samples_per_epoch=10000, balance_train_classes=True,
                                              val_samples_per_epoch=None, test_samples_per_epoch=None):
    """
    Create data loaders with epoch-based downsampling for dual-person architecture
    
    Args:
        data_path: Path to the dataset directory
        batch_size: Batch size for data loaders
        num_workers: Number of data loading workers
        image_size: Target image size for resizing individual person crops
        use_pose: Whether to include pose information
        crop_padding: Padding around person bounding boxes
        min_person_size: Minimum size for person crops
        train_samples_per_epoch: Number of training samples to use per epoch
        balance_train_classes: Whether to maintain class balance in training set
        val_samples_per_epoch: Number of validation samples to use per epoch (None = no downsampling)
        test_samples_per_epoch: Number of test samples to use per epoch (None = no downsampling)
    
    Returns:
        train_loader, val_loader, test_loader, interaction_labels
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets using the downsampling-enabled version
    train_dataset = DownsamplingDualPersonJRDBDataset(
        data_path, split='train', transform=train_transform, 
        image_size=image_size, use_pose=use_pose,
        crop_padding=crop_padding, min_person_size=min_person_size
    )
    
    # Validation and test datasets - can be downsampled or full size
    val_dataset = DownsamplingDualPersonJRDBDataset(
        data_path, split='val', transform=val_test_transform, 
        image_size=image_size, use_pose=use_pose,
        crop_padding=crop_padding, min_person_size=min_person_size
    )
    
    test_dataset = DownsamplingDualPersonJRDBDataset(
        data_path, split='test', transform=val_test_transform, 
        image_size=image_size, use_pose=use_pose,
        crop_padding=crop_padding, min_person_size=min_person_size
    )
    
    # Create samplers for all datasets
    # Handle train_samples_per_epoch=0 as "use all data"
    if train_samples_per_epoch == 0:
        train_sampler = None  # Use all training data without downsampling
        effective_train_samples = len(train_dataset)
        print(f"Using all training data: {effective_train_samples} samples")
    else:
        train_sampler = EpochDownsampleSampler(
            train_dataset, 
            samples_per_epoch=train_samples_per_epoch,
            balance_classes=balance_train_classes
        )
        effective_train_samples = train_samples_per_epoch
    
    # Create validation sampler if downsampling is enabled
    if val_samples_per_epoch is not None:
        val_sampler = EpochDownsampleSampler(
            val_dataset,
            samples_per_epoch=val_samples_per_epoch,
            balance_classes=True  # Keep balance for validation
        )
    else:
        val_sampler = None
    
    # Create test sampler if downsampling is enabled
    if test_samples_per_epoch is not None:
        test_sampler = EpochDownsampleSampler(
            test_dataset,
            samples_per_epoch=test_samples_per_epoch,
            balance_classes=True  # Keep balance for test
        )
    else:
        test_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,  # Use custom sampler or None for full dataset
        shuffle=True if train_sampler is None else False,  # Shuffle only when not using sampler
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,  # Use sampler if downsampling, otherwise None
        shuffle=False if val_sampler else False,  # Don't shuffle if using sampler
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,  # Use sampler if downsampling, otherwise None
        shuffle=False if test_sampler else False,  # Don't shuffle if using sampler
        num_workers=num_workers, 
        pin_memory=True
    )
    
    # Get interaction labels from train dataset
    interaction_labels = getattr(train_dataset, 'actual_interaction_labels', 
                                ['no_interaction', 'walking_together', 'standing_together', 
                                 'conversation', 'sitting_together', 'others'])
    
    print(f"Dual-Person Downsampling Data Loaders Created:")
    print(f"  Original train dataset: {len(train_dataset)} samples")
    print(f"  Train samples per epoch: {effective_train_samples} samples")
    print(f"  Original validation dataset: {len(val_dataset)} samples")
    print(f"  Val samples per epoch: {val_samples_per_epoch or 'Full dataset'}")
    print(f"  Original test dataset: {len(test_dataset)} samples")
    print(f"  Test samples per epoch: {test_samples_per_epoch or 'Full dataset'}")
    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Val batches per epoch: {len(val_loader)}")
    print(f"  Test batches per epoch: {len(test_loader)}")
    print(f"  Class balancing enabled: {balance_train_classes if train_sampler else 'Natural distribution'}")
    
    return train_loader, val_loader, test_loader, interaction_labels


def print_dual_person_dataset_statistics(dataset, name="Dataset"):
    """Print statistics for a dual-person dataset"""
    if not hasattr(dataset, 'samples'):
        print(f"{name}: No samples available")
        return
    
    # Count samples by class
    stage1_counts = Counter()
    stage2_counts = Counter()
    
    for sample in dataset.samples:
        # Stage 1 (binary)
        stage1_label = sample['has_interaction']
        stage1_counts[stage1_label] += 1
        
        # Stage 2 (interaction types) - only for positive samples
        if stage1_label == 1 and 'interaction_label' in sample:
            stage2_label = sample['interaction_label']
            stage2_counts[stage2_label] += 1
    
    print(f"\n{name} Statistics:")
    print(f"  Total samples: {len(dataset.samples)}")
    print(f"  Stage 1 distribution:")
    print(f"    No interaction (0): {stage1_counts[0]}")
    print(f"    Has interaction (1): {stage1_counts[1]}")
    
    if stage2_counts:
        print(f"  Stage 2 distribution (interaction types):")
        interaction_labels = getattr(dataset, 'interaction_labels', 
                                   ['walking_together', 'standing_together', 'conversation', 'sitting_together', 'others'])
        for i, label in enumerate(interaction_labels):
            if i in stage2_counts:
                print(f"    {label} ({i}): {stage2_counts[i]}")


# Test function for the dual-person downsampling dataset
def test_dual_person_downsampling_dataset():
    """Test the dual-person downsampling functionality"""
    print("Testing Dual-Person Downsampling Dataset...")
    
    data_path = 'D:/1data/imagedata'
    
    # Create downsampling data loaders
    train_loader, val_loader, test_loader, interaction_labels = get_dual_person_downsampling_data_loaders(
        data_path=data_path,
        batch_size=8,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        train_samples_per_epoch=500,  # Small number for testing
        val_samples_per_epoch=200,    # Test validation downsampling
        test_samples_per_epoch=100,   # Test test downsampling
        balance_train_classes=True
    )
    
    print(f"\nData loaders created successfully!")
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")  
    print(f"Test loader: {len(test_loader)} batches")
    
    # Test multiple epochs to verify different sampling
    print(f"\nTesting dual-person epoch sampling (first 2 batches per epoch):")
    for epoch in range(3):
        print(f"\n--- Epoch {epoch + 1} ---")
        train_loader.sampler.dataset.set_epoch(epoch)  # Set epoch for logging
        
        batch_count = 0
        pos_count = 0
        neg_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            person_A_images = batch['person_A_image']
            person_B_images = batch['person_B_image']
            stage1_labels = batch['stage1_label']
            
            batch_pos = (stage1_labels == 1).sum().item()
            batch_neg = (stage1_labels == 0).sum().item()
            pos_count += batch_pos
            neg_count += batch_neg
            batch_count += 1
            
            if batch_idx < 2:  # Print first 2 batches
                print(f"  Batch {batch_idx}: Person A: {person_A_images.shape}, Person B: {person_B_images.shape}")
                print(f"    {batch_pos} positive, {batch_neg} negative samples")
                print(f"    Person IDs: A={batch['person_A_id'][:3]}, B={batch['person_B_id'][:3]}")
            
            if batch_count >= 5:  # Sample first 5 batches for speed
                break
        
        print(f"  First 5 batches total: {pos_count} positive, {neg_count} negative")


if __name__ == '__main__':
    test_dual_person_downsampling_dataset()