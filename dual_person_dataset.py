import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from collections import defaultdict, Counter
import cv2
import random


class DualPersonJRDBInteractionDataset(Dataset):
    """
    Dataset for dual-person interaction detection with individual person cropping
    
    Key improvements:
    1. Crops each person individually instead of combined region
    2. Returns separate images for person A and person B
    3. Better handling of crowded scenes
    4. Maintains person identity and features
    """
    
    def __init__(self, data_path, split='train', transform=None, image_size=(224, 224), 
                 use_pose=False, crop_padding=20, min_person_size=32):
        """
        Args:
            data_path: Path to the dataset directory
            split: 'train', 'val', or 'test'
            transform: Image transformations to apply to each person crop
            image_size: Target image size for resizing individual person crops
            use_pose: Whether to include pose information
            crop_padding: Padding around person bounding boxes
            min_person_size: Minimum size for person crops (filter out tiny boxes)
        """
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self.use_pose = use_pose
        self.crop_padding = crop_padding
        self.min_person_size = min_person_size
        
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
        """Load and process the dataset for dual-person architecture"""
        samples = []
        selected_scenes = self._get_scene_splits()
        
        print(f"Loading {len(selected_scenes)} scenes for {self.split} split")
        if len(selected_scenes) == 0:
            print("No scenes found!")
            return []
        
        interaction_counts = Counter()
        valid_samples = 0
        filtered_samples = 0
        
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
            scene_samples, scene_valid, scene_filtered = self._process_scene(
                scene_name, social_data, pose_data, interaction_counts
            )
            samples.extend(scene_samples)
            valid_samples += scene_valid
            filtered_samples += scene_filtered
        
        print(f"Dataset loading summary:")
        print(f"  Valid samples: {valid_samples}")
        print(f"  Filtered samples: {filtered_samples}")
        print(f"  Original interaction distribution: {dict(interaction_counts)}")
        
        # Filter and map interactions to top-4 + others
        filtered_samples = self._filter_and_map_interactions(samples, interaction_counts)
        
        return filtered_samples
    
    def _process_scene(self, scene_name, social_data, pose_data, interaction_counts):
        """Process all frames in a scene for dual-person architecture"""
        samples = []
        valid_count = 0
        filtered_count = 0
        
        # Access the labels dictionary in social data
        if 'labels' not in social_data:
            print(f"No 'labels' key found in social data for scene {scene_name}")
            return samples, valid_count, filtered_count
        
        labels_data = social_data['labels']
        
        for frame_name, frame_annotations in labels_data.items():
            # Build full image path
            image_path = os.path.join(self.images_dir, scene_name, frame_name)
            
            if not os.path.exists(image_path):
                continue
            
            # Extract interaction samples from this frame
            frame_samples, frame_valid, frame_filtered = self._extract_dual_person_interactions(
                image_path, frame_annotations, pose_data, frame_name, interaction_counts
            )
            samples.extend(frame_samples)
            valid_count += frame_valid
            filtered_count += frame_filtered
        
        return samples, valid_count, filtered_count
    
    def _extract_dual_person_interactions(self, image_path, frame_annotations, pose_data, frame_name, interaction_counts):
        """Extract dual-person interaction samples from a single frame"""
        samples = []
        valid_count = 0
        filtered_count = 0
        
        # Build pedestrian dictionary for quick lookup
        pedestrians = {}
        for person_data in frame_annotations:
            if 'label_id' in person_data and 'box' in person_data:
                ped_id = person_data['label_id']
                pedestrians[ped_id] = person_data
        
        # Extract positive interactions (person pairs with interactions)
        for person_data in frame_annotations:
            if 'H-interaction' not in person_data:
                continue
            
            person_id = person_data['label_id']
            person_box = person_data['box']
            
            # Check if person box is valid
            if not self._is_valid_person_box(person_box):
                filtered_count += 1
                continue
            
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
                
                # Check if pair box is valid
                if not self._is_valid_person_box(pair_box):
                    filtered_count += 1
                    continue
                
                # Create positive sample (has interaction)
                sample = {
                    'image_path': image_path,
                    'person_A_box': person_box,  # Individual person A box
                    'person_B_box': pair_box,    # Individual person B box
                    'person_A_id': person_id,
                    'person_B_id': pair_id,
                    'interaction': interaction_type,
                    'has_interaction': 1,
                    'frame_name': frame_name
                }
                
                # Add pose data if available
                if pose_data is not None:
                    person_pose = self._get_pose_for_person(pose_data, frame_name, person_id)
                    pair_pose = self._get_pose_for_person(pose_data, frame_name, pair_id)
                    sample['person_A_pose'] = person_pose
                    sample['person_B_pose'] = pair_pose
                
                samples.append(sample)
                interaction_counts[interaction_type] += 1
                valid_count += 1
        
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
        
        # Generate negative samples with valid box filtering
        negative_samples_per_frame = min(3, len(all_person_ids) // 2)
        generated_negatives = 0
        
        for i, person1_id in enumerate(all_person_ids):
            if generated_negatives >= negative_samples_per_frame:
                break
                
            person1_box = pedestrians[person1_id]['box']
            if not self._is_valid_person_box(person1_box):
                continue
                
            for person2_id in all_person_ids[i+1:]:
                if generated_negatives >= negative_samples_per_frame:
                    break
                
                person2_box = pedestrians[person2_id]['box']
                if not self._is_valid_person_box(person2_box):
                    continue
                
                pair = tuple(sorted([person1_id, person2_id]))
                if pair not in interacting_pairs:
                    sample = {
                        'image_path': image_path,
                        'person_A_box': person1_box,
                        'person_B_box': person2_box,
                        'person_A_id': person1_id,
                        'person_B_id': person2_id,
                        'interaction': 'no_interaction',
                        'has_interaction': 0,
                        'frame_name': frame_name
                    }
                    
                    # Add pose data if available
                    if pose_data is not None:
                        person_pose = self._get_pose_for_person(pose_data, frame_name, person1_id)
                        pair_pose = self._get_pose_for_person(pose_data, frame_name, person2_id)
                        sample['person_A_pose'] = person_pose
                        sample['person_B_pose'] = pair_pose
                    
                    samples.append(sample)
                    interaction_counts['no_interaction'] += 1
                    generated_negatives += 1
                    valid_count += 1
        
        return samples, valid_count, filtered_count
    
    def _is_valid_person_box(self, box):
        """Check if a person bounding box is valid for cropping"""
        x, y, width, height = box
        
        # Check minimum size requirements
        if width < self.min_person_size or height < self.min_person_size:
            return False
        
        # Check if coordinates are reasonable
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return False
        
        return True
    
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
                # Map to 'others' category
                sample_copy = sample.copy()
                sample_copy['interaction_label'] = 4  # Others
                sample_copy['mapped_interaction'] = 'others'
                filtered_samples.append(sample_copy)
                new_interaction_counts['others'] += 1
        
        # Store actual labels found in this dataset
        self.actual_interaction_labels = ['no_interaction'] + top_interaction_types + ['others']
        
        print(f"Final interaction distribution: {dict(new_interaction_counts)}")
        return filtered_samples
    
    def _print_statistics(self):
        """Print dataset statistics"""
        if not self.samples:
            print("No samples loaded")
            return
        
        print(f"\n{self.split.upper()} Dataset Statistics:")
        print(f"  Total samples: {len(self.samples)}")
        
        # Stage 1 statistics
        stage1_counts = Counter()
        stage2_counts = Counter()
        
        for sample in self.samples:
            stage1_counts[sample['has_interaction']] += 1
            if sample['has_interaction'] == 1 and 'interaction_label' in sample:
                stage2_counts[sample['interaction_label']] += 1
        
        print(f"  Stage 1 (Binary interaction detection):")
        print(f"    No interaction (0): {stage1_counts[0]}")
        print(f"    Has interaction (1): {stage1_counts[1]}")
        print(f"    Ratio (pos:neg): {stage1_counts[1]/(stage1_counts[0]+1):.2f}")
        
        if stage2_counts:
            print(f"  Stage 2 (Interaction type classification):")
            for i, label in enumerate(self.interaction_labels):
                if i in stage2_counts:
                    print(f"    {label} ({i}): {stage2_counts[i]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample with individual person crops"""
        sample = self.samples[idx]
        
        # Load the full image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return dummy images
            person_A_image = Image.new('RGB', self.image_size, (0, 0, 0))
            person_B_image = Image.new('RGB', self.image_size, (0, 0, 0))
        else:
            # Crop individual person regions
            person_A_image = self._crop_person(image, sample['person_A_box'])
            person_B_image = self._crop_person(image, sample['person_B_box'])
        
        # Apply transforms to both person images
        if self.transform:
            person_A_image = self.transform(person_A_image)
            person_B_image = self.transform(person_B_image)
        
        # Prepare labels
        stage1_label = sample['has_interaction']
        
        if sample['has_interaction'] == 1 and 'interaction_label' in sample:
            stage2_label = sample['interaction_label']
        else:
            stage2_label = -1  # Invalid label for no interaction cases
        
        result = {
            'person_A_image': person_A_image,
            'person_B_image': person_B_image,
            'stage1_label': torch.tensor(stage1_label, dtype=torch.long),
            'stage2_label': torch.tensor(stage2_label, dtype=torch.long),
            'image_path': sample['image_path'],
            'person_A_box': torch.tensor(sample['person_A_box'], dtype=torch.float32),
            'person_B_box': torch.tensor(sample['person_B_box'], dtype=torch.float32),
            'person_A_id': sample['person_A_id'],
            'person_B_id': sample['person_B_id']
        }
        
        # Add pose information if available
        if self.use_pose:
            person_A_pose = sample.get('person_A_pose', [0] * 51)  # 17 * 3
            person_B_pose = sample.get('person_B_pose', [0] * 51)  # 17 * 3
            
            if person_A_pose is None:
                person_A_pose = [0] * 51
            if person_B_pose is None:
                person_B_pose = [0] * 51
            
            result['person_A_pose'] = torch.tensor(person_A_pose, dtype=torch.float32)
            result['person_B_pose'] = torch.tensor(person_B_pose, dtype=torch.float32)
        
        return result
    
    def _crop_person(self, image, person_box):
        """Crop individual person from the full image"""
        x, y, width, height = person_box
        
        # Add padding
        img_width, img_height = image.size
        x1 = max(0, x - self.crop_padding)
        y1 = max(0, y - self.crop_padding)
        x2 = min(img_width, x + width + self.crop_padding)
        y2 = min(img_height, y + height + self.crop_padding)
        
        # Ensure valid crop region
        if x2 <= x1 or y2 <= y1:
            # Return a small crop from center if invalid
            center_x, center_y = img_width // 2, img_height // 2
            crop_size = min(100, img_width // 4, img_height // 4)
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(img_width, center_x + crop_size // 2)
            y2 = min(img_height, center_y + crop_size // 2)
        
        cropped_image = image.crop((x1, y1, x2, y2))
        
        # Ensure minimum size
        if cropped_image.size[0] < 32 or cropped_image.size[1] < 32:
            cropped_image = cropped_image.resize((64, 64), Image.LANCZOS)
        
        return cropped_image


def get_dual_person_data_loaders(data_path, batch_size=16, num_workers=4, image_size=(224, 224), use_pose=False):
    """
    Create data loaders for dual-person train, validation, and test sets
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
    train_dataset = DualPersonJRDBInteractionDataset(
        data_path, split='train', transform=train_transform, 
        image_size=image_size, use_pose=use_pose
    )
    val_dataset = DualPersonJRDBInteractionDataset(
        data_path, split='val', transform=val_test_transform, 
        image_size=image_size, use_pose=use_pose
    )
    test_dataset = DualPersonJRDBInteractionDataset(
        data_path, split='test', transform=val_test_transform, 
        image_size=image_size, use_pose=use_pose
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    # Get interaction labels from train dataset
    interaction_labels = getattr(train_dataset, 'actual_interaction_labels', 
                                ['no_interaction', 'walking_together', 'standing_together', 
                                 'conversation', 'sitting_together', 'others'])
    
    print(f"Dual-Person Data Loaders Created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Interaction labels: {interaction_labels}")
    
    return train_loader, val_loader, test_loader, interaction_labels


if __name__ == '__main__':
    # Test the dual-person dataset
    print("Testing Dual-Person Dataset...")
    
    data_path = 'D:/1data/imagedata'
    
    try:
        # Create a small test dataset
        train_loader, val_loader, test_loader, interaction_labels = get_dual_person_data_loaders(
            data_path=data_path,
            batch_size=4,
            num_workers=0,
            image_size=(224, 224),
            use_pose=False
        )
        
        print(f"\nTesting data loader...")
        
        # Test first batch
        for batch_idx, batch in enumerate(train_loader):
            person_A_images = batch['person_A_image']
            person_B_images = batch['person_B_image']
            stage1_labels = batch['stage1_label']
            stage2_labels = batch['stage2_label']
            
            print(f"  Batch {batch_idx}:")
            print(f"    Person A images: {person_A_images.shape}")
            print(f"    Person B images: {person_B_images.shape}")
            print(f"    Stage 1 labels: {stage1_labels}")
            print(f"    Stage 2 labels: {stage2_labels}")
            print(f"    Person A IDs: {batch['person_A_id']}")
            print(f"    Person B IDs: {batch['person_B_id']}")
            
            if batch_idx >= 2:  # Test first 3 batches
                break
        
        print(f"\n✓ Dual-person dataset working correctly!")
        
    except Exception as e:
        print(f"✗ Error testing dual-person dataset: {e}")
        import traceback
        traceback.print_exc()