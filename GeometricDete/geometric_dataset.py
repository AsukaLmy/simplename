import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_features import extract_geometric_features, extract_causal_motion_features, compute_scene_context
from temporal_buffer import CausalTemporalBuffer, TemporalPairManager

def _get_frame_id_sort_key(sample):
    """Helper function to get frame_id for sorting (pickle-friendly)"""
    return sample['frame_id']


class GeometricDualPersonDataset(Dataset):
    """
    Dataset for geometric dual-person interaction detection
    Focuses on geometric features rather than visual features
    """
    
    def __init__(self, data_path, split='train', history_length=5, 
                 use_temporal=True, use_scene_context=True):
        """
        Args:
            data_path: Path to dataset root directory
            split: 'train', 'val', or 'test'
            history_length: Number of historical frames to use
            use_temporal: Whether to use temporal features
            use_scene_context: Whether to use scene context features
        """
        self.data_path = data_path
        self.split = split
        self.history_length = history_length
        self.use_temporal = use_temporal
        self.use_scene_context = use_scene_context
        
        # Initialize temporal manager
        if use_temporal:
            self.temporal_manager = TemporalPairManager(history_length=history_length)
        else:
            self.temporal_manager = None
        
        # Load data
        self.samples = []
        self.scene_data = {}  # For scene context computation
        
        self._load_data()
        self._precompute_scene_context()
        
        print(f"GeometricDualPersonDataset loaded: {len(self.samples)} samples ({split})")
        print(f"  Temporal features: {use_temporal}")
        print(f"  Scene context: {use_scene_context}")
    
    def _load_data(self):
        """Load geometric interaction data from JRDB format"""
        # JRDB format: separate JSON files for each scene
        social_labels_dir = os.path.join(self.data_path, 'labels', 'labels_2d_activity_social_stitched')
        
        if not os.path.exists(social_labels_dir):
            raise FileNotFoundError(f"Social labels directory not found: {social_labels_dir}")
        
        # Get all scene files
        scene_files = [f for f in os.listdir(social_labels_dir) if f.endswith('.json')]
        
        # Split scenes for train/val/test
        scene_files.sort()  # Ensure consistent ordering
        total_scenes = len(scene_files)
        
        if self.split == 'train':
            selected_files = scene_files[:int(0.7 * total_scenes)]
        elif self.split == 'val':
            selected_files = scene_files[int(0.7 * total_scenes):int(0.85 * total_scenes)]
        else:  # test
            selected_files = scene_files[int(0.85 * total_scenes):]
        
        print(f"Loading {len(selected_files)} scenes for {self.split} split")
        
        # Load data from selected scenes
        all_social_data = {}
        for scene_file in selected_files:
            scene_path = os.path.join(social_labels_dir, scene_file)
            scene_name = os.path.splitext(scene_file)[0]
            
            try:
                with open(scene_path, 'r') as f:
                    scene_data = json.load(f)
                all_social_data[scene_name] = scene_data
            except Exception as e:
                print(f"Error loading scene {scene_file}: {e}")
                continue
        
        # Process social annotations to extract geometric pairs
        frame_count = 0
        
        for scene_name, scene_data in all_social_data.items():
            for image_name, annotations in scene_data.get('labels', {}).items():
                # Create unique frame_id combining scene and image
                frame_id = f"{scene_name}_{self._extract_frame_id(image_name)}"
                
                # Collect all person boxes in this frame for scene context
                all_boxes = []
                person_dict = {}
                
                for ann in annotations:
                    person_id = ann.get('label_id', '')
                    if person_id.startswith('pedestrian:'):
                        pid = int(person_id.split(':')[1])
                        box = ann.get('box', [0, 0, 100, 100])
                        all_boxes.append(box)
                        person_dict[pid] = {
                            'box': box,
                            'actions': ann.get('action_label', {}),
                            'interactions': ann.get('H-interaction', [])
                        }
                
                # Store scene information
                self.scene_data[frame_id] = {
                    'scene_name': scene_name,
                    'image_name': image_name,
                    'all_boxes': all_boxes,
                    'persons': person_dict
                }
                
                # Generate positive samples (has interaction)
                for ann in annotations:
                    person_id = ann.get('label_id', '')
                    if not person_id.startswith('pedestrian:'):
                        continue
                    
                    person_A_id = int(person_id.split(':')[1])
                    person_A_box = ann.get('box', [0, 0, 100, 100])
                    
                    # Process H-interaction (JRDB format)
                    for interaction in ann.get('H-interaction', []):
                        pair_id = interaction.get('pair', '')
                        if pair_id.startswith('pedestrian:'):
                            person_B_id = int(pair_id.split(':')[1])
                            
                            if person_B_id in person_dict:
                                person_B_box = person_dict[person_B_id]['box']
                                interaction_labels = interaction.get('inter_labels', {})
                                
                                # Create positive sample
                                sample = {
                                    'frame_id': frame_id,
                                    'scene_name': scene_name,
                                    'image_name': image_name,
                                    'person_A_id': person_A_id,
                                    'person_B_id': person_B_id,
                                    'person_A_box': person_A_box,
                                    'person_B_box': person_B_box,
                                    'has_interaction': 1,
                                    'interaction_labels': interaction_labels,
                                    'sample_type': 'positive'
                                }
                                self.samples.append(sample)
                
                # Generate negative samples (no interaction)
                person_ids = list(person_dict.keys())
                if len(person_ids) >= 2:
                    # Find pairs without interactions
                    interacting_pairs = set()
                    for ann in annotations:
                        person_id = ann.get('label_id', '')
                        if person_id.startswith('pedestrian:'):
                            person_A_id = int(person_id.split(':')[1])
                            for interaction in ann.get('H-interaction', []):
                                pair_id = interaction.get('pair', '')
                                if pair_id.startswith('pedestrian:'):
                                    person_B_id = int(pair_id.split(':')[1])
                                    interacting_pairs.add(tuple(sorted([person_A_id, person_B_id])))
                    
                    # Generate negative samples
                    neg_count = 0
                    max_neg_per_frame = min(len(person_ids) * 2, 10)
                    
                    for i, person_A_id in enumerate(person_ids):
                        for person_B_id in person_ids[i+1:]:
                            pair = tuple(sorted([person_A_id, person_B_id]))
                            if pair not in interacting_pairs and neg_count < max_neg_per_frame:
                                sample = {
                                    'frame_id': frame_id,
                                    'scene_name': scene_name,
                                    'image_name': image_name,
                                    'person_A_id': person_A_id,
                                    'person_B_id': person_B_id,
                                    'person_A_box': person_dict[person_A_id]['box'],
                                    'person_B_box': person_dict[person_B_id]['box'],
                                    'has_interaction': 0,
                                    'interaction_labels': {},
                                    'sample_type': 'negative'
                                }
                                self.samples.append(sample)
                                neg_count += 1
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames, {len(self.samples)} samples")
    
    def _extract_frame_id(self, image_name):
        """Extract frame ID from image name (JRDB format)"""
        # JRDB format: "000000.jpg" -> "000000"
        return os.path.splitext(image_name)[0]
    
    def _precompute_scene_context(self):
        """Precompute scene context for all frames"""
        total_frames = len(self.scene_data)
        print(f"Precomputing scene context for {total_frames} frames...")
        
        processed = 0
        for frame_id, scene_info in self.scene_data.items():
            all_boxes = scene_info['all_boxes']
            if len(all_boxes) > 0:
                # Assume standard image dimensions (can be refined)
                scene_context = compute_scene_context(all_boxes, 640, 480)
                self.scene_data[frame_id]['scene_context'] = scene_context
            else:
                self.scene_data[frame_id]['scene_context'] = torch.tensor([0.0])  # Empty scene
            
            processed += 1
            if processed % 1000 == 0:  # Every 1000 frames
                print(f"  Progress: {processed}/{total_frames} frames ({100*processed/total_frames:.1f}%)")
        
        print(f"Scene context precomputation completed: {total_frames} frames processed")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample with geometric features and optional temporal information
        """
        sample = self.samples[idx]
        
        # Extract basic information
        frame_id = sample['frame_id']
        person_A_id = sample['person_A_id']
        person_B_id = sample['person_B_id']
        person_A_box = torch.tensor(sample['person_A_box'], dtype=torch.float32)
        person_B_box = torch.tensor(sample['person_B_box'], dtype=torch.float32)
        has_interaction = torch.tensor(sample['has_interaction'], dtype=torch.long)
        
        # Extract geometric features
        geometric_features = extract_geometric_features(
            person_A_box, person_B_box, 640, 480
        )
        
        # Prepare result
        result = {
            'geometric_features': geometric_features,
            'stage1_label': has_interaction,
            'person_A_id': person_A_id,
            'person_B_id': person_B_id,
            'frame_id': frame_id,
            'person_A_box': person_A_box,
            'person_B_box': person_B_box
        }
        
        # Add scene context if enabled
        if self.use_scene_context and frame_id in self.scene_data:
            result['scene_context'] = self.scene_data[frame_id]['scene_context']
        else:
            result['scene_context'] = torch.tensor([1.0])  # Default: sparse scene
        
        # Add temporal features if enabled
        if self.use_temporal and self.temporal_manager:
            temporal_features = self.temporal_manager.get_temporal_features(person_A_id, person_B_id)
            
            # Use pair interaction history as the main temporal signal
            result['history_geometric'] = temporal_features['pair_interaction_history']
            result['has_sufficient_history'] = temporal_features['has_sufficient_history']
            
            # Extract motion features from historical data
            if temporal_features['has_sufficient_history']:
                history_data = temporal_features['pair_interaction_history']
                if history_data.size(0) >= 2:  # Need at least 2 time steps
                    motion_features = extract_causal_motion_features(history_data.unsqueeze(0))
                    result['motion_features'] = motion_features.squeeze(0)
                else:
                    result['motion_features'] = torch.zeros(4)
            else:
                result['motion_features'] = torch.zeros(4)
        else:
            result['history_geometric'] = torch.zeros(self.history_length, 7)
            result['has_sufficient_history'] = False
            result['motion_features'] = torch.zeros(4)
        
        return result
    
    def update_temporal_buffer(self):
        """Update temporal buffer with all data (for proper temporal modeling)"""
        if not self.temporal_manager:
            return
        
        print("Updating temporal buffer...")
        
        # Sort samples by frame_id for temporal consistency
        sorted_samples = sorted(self.samples, key=_get_frame_id_sort_key)
        
        # Group by frame
        frames = {}
        for sample in sorted_samples:
            frame_id = sample['frame_id']
            if frame_id not in frames:
                frames[frame_id] = []
            frames[frame_id].append(sample)
        
        # Process frames in order
        for frame_id in sorted(frames.keys()):
            frame_samples = frames[frame_id]
            frame_data = []
            
            for sample in frame_samples:
                geometric_features = extract_geometric_features(
                    torch.tensor(sample['person_A_box']), 
                    torch.tensor(sample['person_B_box']),
                    640, 480
                )
                
                frame_data.append({
                    'person_A_id': sample['person_A_id'],
                    'person_B_id': sample['person_B_id'],
                    'geometric_features': geometric_features
                })
            
            # Update temporal manager
            self.temporal_manager.update_frame(frame_data, frame_id)
        
        print("Temporal buffer updated!")
    
    def get_class_distribution(self):
        """Get class distribution for balancing"""
        positive = sum(1 for s in self.samples if s['has_interaction'] == 1)
        negative = len(self.samples) - positive
        return {'positive': positive, 'negative': negative, 'total': len(self.samples)}


def create_geometric_data_loaders(data_path, batch_size=32, num_workers=4, 
                                history_length=5, use_temporal=False, use_scene_context=True):
    """
    Create data loaders for geometric dual-person interaction detection
    
    Args:
        data_path: Path to dataset root
        batch_size: Batch size for data loaders
        num_workers: Number of data loading workers
        history_length: Number of historical frames
        use_temporal: Whether to use temporal features
        use_scene_context: Whether to use scene context
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = GeometricDualPersonDataset(
        data_path, split='train', history_length=history_length,
        use_temporal=use_temporal, use_scene_context=use_scene_context
    )
    
    val_dataset = GeometricDualPersonDataset(
        data_path, split='val', history_length=history_length,
        use_temporal=use_temporal, use_scene_context=use_scene_context
    )
    
    test_dataset = GeometricDualPersonDataset(
        data_path, split='test', history_length=history_length,
        use_temporal=use_temporal, use_scene_context=use_scene_context
    )
    
    # Update temporal buffers for proper temporal modeling
    if use_temporal:
        print("Initializing temporal modeling...")
        train_dataset.update_temporal_buffer()
        val_dataset.update_temporal_buffer()
        test_dataset.update_temporal_buffer()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Print dataset statistics
    train_dist = train_dataset.get_class_distribution()
    val_dist = val_dataset.get_class_distribution()
    test_dist = test_dataset.get_class_distribution()
    
    print(f"\nDataset Statistics:")
    print(f"Train: {train_dist}")
    print(f"Val: {val_dist}")
    print(f"Test: {test_dist}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    print("Testing GeometricDualPersonDataset...")
    
    # You would need to provide actual data path
    # data_path = 'D:/1data/imagedata'
    
    # For testing with dummy data
    print("This is a code structure test. Replace with actual data_path for real testing.")
    print("Dataset implementation completed!")