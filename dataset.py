import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from collections import defaultdict, Counter
import cv2


class JRDBInteractionDataset(Dataset):
    """
    Dataset for human interaction detection using JRDB format
    Based on the actual JRDB dataset structure with social and pose annotations
    """
    
    def __init__(self, data_path, split='train', transform=None, image_size=(224, 224), use_pose=False):
        """
        Args:
            data_path: Path to the dataset directory (D:/1data/imagedata)
            split: 'train', 'val', or 'test'
            transform: Image transformations
            image_size: Target image size for resizing
            use_pose: Whether to include pose information
        """
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self.use_pose = use_pose
        
        # Paths for different data types
        self.images_dir = os.path.join(data_path, 'images', 'image_stitched')
        self.social_labels_dir = os.path.join(data_path, 'labels', 'labels_2d_activity_social_stitched')
        self.pose_labels_dir = os.path.join(data_path, 'labels', 'labels_2d_pose_stitched_coco')
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # H-interaction mapping based on the statistics from the report
        self.interaction_mapping = {
            'walking together': 0,      # 46.9%
            'standing together': 1,     # 33.7%
            'conversation': 2,          # 8.8%
            'sitting together': 3,      # 7.7%
            # All other 15 types mapped to 4 (others)
        }
        
        self.interaction_labels = [
            'walking_together',
            'standing_together', 
            'conversation',
            'sitting_together',
            'others'
        ]
        
        # Load dataset
        self.samples = self._load_dataset()
        
        # Print dataset statistics
        self._print_statistics()
    
    def _get_scene_splits(self):
        """Split scenes into train/val/test"""
        # Get all available scenes
        scenes = []
        if os.path.exists(self.social_labels_dir):
            for file in os.listdir(self.social_labels_dir):
                if file.endswith('.json'):
                    scene_name = file.replace('.json', '')
                    scenes.append(scene_name)
        
        scenes.sort()  # Ensure consistent splitting
        total_scenes = len(scenes)
        
        # Split scenes: 70% train, 15% val, 15% test
        train_end = int(0.7 * total_scenes)
        val_end = int(0.85 * total_scenes)
        
        if self.split == 'train':
            return scenes[:train_end]
        elif self.split == 'val':
            return scenes[train_end:val_end]
        else:  # test
            return scenes[val_end:]
    
    def _load_dataset(self):
        """Load and process the dataset according to JRDB format"""
        samples = []
        selected_scenes = self._get_scene_splits()
        
        print(f"Loading {len(selected_scenes)} scenes for {self.split} split")
        if len(selected_scenes) == 0:
            print("No scenes found!")
            return []
        
        interaction_counts = Counter()
        
        for scene_name in selected_scenes:
            print(f"Processing scene: {scene_name}")
            
            # Load social annotations
            social_file = os.path.join(self.social_labels_dir, f"{scene_name}.json")
            if not os.path.exists(social_file):
                print(f"Social annotation file not found: {social_file}")
                continue
            
            try:
                with open(social_file, 'r') as f:
                    social_data = json.load(f)
            except Exception as e:
                print(f"Error loading social data from {social_file}: {e}")
                continue
            
            # Load pose annotations if needed
            pose_data = None
            if self.use_pose:
                pose_file = os.path.join(self.pose_labels_dir, f"{scene_name}.json")
                if os.path.exists(pose_file):
                    try:
                        with open(pose_file, 'r') as f:
                            pose_data = json.load(f)
                    except Exception as e:
                        print(f"Error loading pose data from {pose_file}: {e}")
            
            # Process frames in this scene
            scene_samples = self._process_scene(scene_name, social_data, pose_data, interaction_counts)
            samples.extend(scene_samples)
        
        print(f"Original interaction distribution: {dict(interaction_counts)}")
        
        # Filter and map interactions to top-4 + others
        filtered_samples = self._filter_and_map_interactions(samples, interaction_counts)
        
        return filtered_samples
    
    def _process_scene(self, scene_name, social_data, pose_data, interaction_counts):
        """Process all frames in a scene"""
        samples = []
        
        # Access the labels dictionary in social data
        if 'labels' not in social_data:
            print(f"No 'labels' key found in social data for scene {scene_name}")
            return samples
        
        labels_data = social_data['labels']
        
        for frame_name, frame_annotations in labels_data.items():
            # Build full image path
            image_path = os.path.join(self.images_dir, scene_name, frame_name)
            
            if not os.path.exists(image_path):
                continue
            
            # Extract interaction samples from this frame
            frame_samples = self._extract_interactions_from_frame(
                image_path, frame_annotations, pose_data, frame_name, interaction_counts
            )
            samples.extend(frame_samples)
        
        return samples
    
    def _extract_interactions_from_frame(self, image_path, frame_annotations, pose_data, frame_name, interaction_counts):
        """Extract interaction samples from a single frame"""
        samples = []
        
        # Build pedestrian dictionary for quick lookup
        pedestrians = {}
        for person_data in frame_annotations:
            if 'label_id' in person_data and 'box' in person_data:
                ped_id = person_data['label_id']
                pedestrians[ped_id] = person_data
        
        # Extract interactions
        for person_data in frame_annotations:
            if 'H-interaction' not in person_data:
                continue
            
            person_id = person_data['label_id']
            person_box = person_data['box']
            
            # Process each interaction
            for interaction in person_data['H-interaction']:
                if 'pair' not in interaction or 'inter_labels' not in interaction:
                    continue
                
                pair_id = interaction['pair']
                pair_box = interaction.get('box_pair', None)
                inter_labels = interaction['inter_labels']
                
                # Get the interaction type (first one if multiple)
                interaction_type = list(inter_labels.keys())[0] if inter_labels else None
                
                if interaction_type is None:
                    continue
                
                # Ensure we have both boxes
                if pair_box is None:
                    if pair_id in pedestrians:
                        pair_box = pedestrians[pair_id]['box']
                    else:
                        continue
                
                # Create positive sample (has interaction)
                sample = {
                    'image_path': image_path,
                    'bbox1': person_box,
                    'bbox2': pair_box,
                    'interaction': interaction_type,
                    'has_interaction': 1,
                    'person_ids': [person_id, pair_id],
                    'frame_name': frame_name
                }
                
                # Add pose data if available
                if pose_data is not None:
                    person_pose = self._get_pose_for_person(pose_data, frame_name, person_id)
                    pair_pose = self._get_pose_for_person(pose_data, frame_name, pair_id)
                    sample['person_pose'] = person_pose
                    sample['pair_pose'] = pair_pose
                
                samples.append(sample)
                interaction_counts[interaction_type] += 1
        
        # Generate negative samples (no interaction)
        # Take pairs of people who are not interacting
        all_person_ids = list(pedestrians.keys())
        interacting_pairs = set()
        
        # Collect all interacting pairs
        for person_data in frame_annotations:
            if 'H-interaction' in person_data:
                person_id = person_data['label_id']
                for interaction in person_data['H-interaction']:
                    if 'pair' in interaction:
                        pair_id = interaction['pair']
                        # Create canonical pair representation (sorted)
                        pair = tuple(sorted([person_id, pair_id]))
                        interacting_pairs.add(pair)
        
        # Generate negative samples
        negative_samples_per_frame = min(3, len(all_person_ids) // 2)  # Limit negative samples
        generated_negatives = 0
        
        for i, person1_id in enumerate(all_person_ids):
            if generated_negatives >= negative_samples_per_frame:
                break
                
            for person2_id in all_person_ids[i+1:]:
                if generated_negatives >= negative_samples_per_frame:
                    break
                
                pair = tuple(sorted([person1_id, person2_id]))
                if pair not in interacting_pairs:
                    sample = {
                        'image_path': image_path,
                        'bbox1': pedestrians[person1_id]['box'],
                        'bbox2': pedestrians[person2_id]['box'],
                        'interaction': 'no_interaction',
                        'has_interaction': 0,
                        'person_ids': [person1_id, person2_id],
                        'frame_name': frame_name
                    }
                    
                    # Add pose data if available
                    if pose_data is not None:
                        person_pose = self._get_pose_for_person(pose_data, frame_name, person1_id)
                        pair_pose = self._get_pose_for_person(pose_data, frame_name, person2_id)
                        sample['person_pose'] = person_pose
                        sample['pair_pose'] = pair_pose
                    
                    samples.append(sample)
                    interaction_counts['no_interaction'] += 1
                    generated_negatives += 1
        
        return samples
    
    def _get_pose_for_person(self, pose_data, frame_name, person_id):
        """Get pose data for a specific person in a frame"""
        if pose_data is None:
            return None
        
        # Extract track_id from person_id (pedestrian:X -> X)
        try:
            track_id = int(person_id.split(':')[1])
        except:
            return None
        
        # Find the image ID for this frame
        frame_image_id = None
        for img in pose_data.get('images', []):
            img_filename = os.path.basename(img['file_name'])
            if img_filename == frame_name:
                frame_image_id = img['id']
                break
        
        if frame_image_id is None:
            return None
        
        # Find pose annotation for this person
        for annotation in pose_data.get('annotations', []):
            if (annotation.get('image_id') == frame_image_id and 
                annotation.get('track_id') == track_id):
                return annotation.get('keypoints', None)
        
        return None
    
    def _filter_and_map_interactions(self, samples, interaction_counts):
        """Map interactions to top-4 + others categories"""
        # Get interaction-only counts (exclude 'no_interaction')
        interaction_only_counts = {k: v for k, v in interaction_counts.items() 
                                 if k != 'no_interaction'}
        
        # Get top 4 interactions
        top_interactions = Counter(interaction_only_counts).most_common(4)
        top_interaction_types = [item[0] for item in top_interactions]
        
        print(f"Top 4 interactions found in data: {top_interaction_types}")
        
        # Update the mapping with actual top interactions found
        actual_mapping = {}
        for i, interaction_type in enumerate(top_interaction_types):
            actual_mapping[interaction_type] = i
        
        # Map remaining interactions to 'others' (label 4)
        filtered_samples = []
        new_interaction_counts = Counter()
        
        for sample in samples:
            original_interaction = sample['interaction']
            
            if original_interaction == 'no_interaction':
                # Keep no interaction samples as is
                sample_copy = sample.copy()
                filtered_samples.append(sample_copy)
                new_interaction_counts['no_interaction'] += 1
            elif original_interaction in actual_mapping:
                # Map to top-4 categories
                sample_copy = sample.copy()
                sample_copy['interaction_label'] = actual_mapping[original_interaction]
                sample_copy['mapped_interaction'] = top_interaction_types[actual_mapping[original_interaction]]
                filtered_samples.append(sample_copy)
                new_interaction_counts[original_interaction] += 1
            else:
                # Map to 'others'
                sample_copy = sample.copy()
                sample_copy['interaction'] = 'others'
                sample_copy['interaction_label'] = 4  # Others category
                sample_copy['mapped_interaction'] = 'others'
                filtered_samples.append(sample_copy)
                new_interaction_counts['others'] += 1
        
        print(f"Filtered interaction distribution: {dict(new_interaction_counts)}")
        
        # Update interaction labels with actual found interactions
        self.actual_top_interactions = top_interaction_types
        self.actual_interaction_labels = top_interaction_types + ['others']
        
        return filtered_samples
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"\n=== JRDB Dataset Statistics ({self.split}) ===")
        print(f"Total samples: {len(self.samples)}")
        
        if len(self.samples) == 0:
            print("No samples loaded!")
            return
        
        # Count by interaction type
        interaction_counts = Counter()
        has_interaction_counts = Counter()
        
        for sample in self.samples:
            has_interaction_counts[sample['has_interaction']] += 1
            
            if sample['has_interaction'] == 0:
                interaction_counts['no_interaction'] += 1
            else:
                if 'interaction_label' in sample:
                    label_idx = sample['interaction_label']
                    if hasattr(self, 'actual_interaction_labels') and label_idx < len(self.actual_interaction_labels):
                        label_name = self.actual_interaction_labels[label_idx]
                    else:
                        label_name = f"label_{label_idx}"
                    interaction_counts[label_name] += 1
                else:
                    interaction_counts[sample['interaction']] += 1
        
        print(f"Interaction distribution: {dict(interaction_counts)}")
        print(f"Has interaction: {dict(has_interaction_counts)}")
        
        if hasattr(self, 'actual_interaction_labels'):
            print(f"Actual interaction labels: {self.actual_interaction_labels}")
        
        print("=" * 50)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a dummy black image
            image = Image.new('RGB', self.image_size, (0, 0, 0))
        
        # Get bounding boxes
        bbox1 = sample['bbox1']  # [x, y, width, height]
        bbox2 = sample['bbox2']
        
        # Calculate combined bounding box that contains both people
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        # Add padding
        padding = 20
        img_width, img_height = image.size
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        
        # Ensure valid crop region
        if x2 <= x1 or y2 <= y1:
            # Use full image if crop region is invalid
            cropped_image = image
        else:
            cropped_image = image.crop((x1, y1, x2, y2))
        
        # Apply transforms
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        # Prepare labels
        stage1_label = sample['has_interaction']
        
        if sample['has_interaction'] == 1 and 'interaction_label' in sample:
            stage2_label = sample['interaction_label']
        else:
            stage2_label = -1  # Invalid label for no interaction cases
        
        result = {
            'image': cropped_image,
            'stage1_label': torch.tensor(stage1_label, dtype=torch.long),
            'stage2_label': torch.tensor(stage2_label, dtype=torch.long),
            'image_path': sample['image_path'],
            'bbox1': torch.tensor(bbox1, dtype=torch.float32),
            'bbox2': torch.tensor(bbox2, dtype=torch.float32)
        }
        
        # Add pose information if available
        if self.use_pose and 'person_pose' in sample and 'pair_pose' in sample:
            person_pose = sample['person_pose'] if sample['person_pose'] is not None else [0] * 51  # 17 * 3
            pair_pose = sample['pair_pose'] if sample['pair_pose'] is not None else [0] * 51
            
            result['person_pose'] = torch.tensor(person_pose, dtype=torch.float32)
            result['pair_pose'] = torch.tensor(pair_pose, dtype=torch.float32)
        
        return result


def get_data_loaders(data_path, batch_size=16, num_workers=4, image_size=(224, 224), use_pose=False):
    """
    Create data loaders for train, validation, and test sets
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
    
    # Create datasets
    train_dataset = JRDBInteractionDataset(data_path, split='train', transform=train_transform, 
                                          image_size=image_size, use_pose=use_pose)
    val_dataset = JRDBInteractionDataset(data_path, split='val', transform=val_test_transform, 
                                        image_size=image_size, use_pose=use_pose)
    test_dataset = JRDBInteractionDataset(data_path, split='test', transform=val_test_transform, 
                                         image_size=image_size, use_pose=use_pose)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    # Get interaction labels from train dataset
    interaction_labels = getattr(train_dataset, 'actual_interaction_labels', 
                                ['walking_together', 'standing_together', 'conversation', 'sitting_together', 'others'])
    
    return train_loader, val_loader, test_loader, interaction_labels


if __name__ == '__main__':
    # Test the dataset
    data_path = 'D:/1data/imagedata'  # Path from the format report
    
    print("Testing JRDB dataset loading...")
    
    try:
        # Test basic dataset loading
        dataset = JRDBInteractionDataset(data_path, split='train', use_pose=False)
        
        if len(dataset) > 0:
            print(f"Dataset loaded successfully with {len(dataset)} samples")
            
            # Test a sample
            sample = dataset[0]
            print(f"Sample image shape: {sample['image'].shape}")
            print(f"Stage 1 label: {sample['stage1_label']}")
            print(f"Stage 2 label: {sample['stage2_label']}")
            
            # Test data loader
            print("\nTesting data loaders...")
            train_loader, val_loader, test_loader, labels = get_data_loaders(
                data_path, batch_size=4, num_workers=0  # Use 0 workers for testing
            )
            
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
            print(f"Interaction labels: {labels}")
            
            # Test one batch
            if len(train_loader) > 0:
                batch = next(iter(train_loader))
                print(f"Batch image shape: {batch['image'].shape}")
                print(f"Batch stage1 labels: {batch['stage1_label']}")
                print(f"Batch stage2 labels: {batch['stage2_label']}")
        else:
            print("No samples found in dataset. Check data paths and file structure.")
            
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()