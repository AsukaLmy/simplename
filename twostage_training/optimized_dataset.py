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
import math


class OptimizedJRDBInteractionDataset(Dataset):
    """
    Optimized dataset for human interaction detection with distance-based pairing
    and sample balancing for improved training efficiency
    """
    
    def __init__(self, data_path, split='train', transform=None, image_size=(224, 224), 
                 use_pose=False, distance_multiplier=3.0, max_negatives_per_frame=5,
                 stage1_balance_ratio=1.0, stage2_balance_strategy='oversample',
                 use_group_sampling=True, max_group_samples_ratio=1.0):
        """
        Args:
            data_path: Path to the dataset directory (D:/1data/imagedata)
            split: 'train', 'val', or 'test'
            transform: Image transformations
            image_size: Target image size for resizing
            use_pose: Whether to include pose information
            distance_multiplier: Maximum distance for pairing (in multiples of box width)
            max_negatives_per_frame: Maximum negative samples per frame
            stage1_balance_ratio: Negative to positive ratio for stage 1
            stage2_balance_strategy: 'oversample', 'undersample', or 'weighted'
            use_group_sampling: Whether to use group-based positive sampling
            max_group_samples_ratio: Maximum samples per group as ratio of group size (1.0 = n samples for n people)
        """
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self.use_pose = use_pose
        self.distance_multiplier = distance_multiplier
        self.max_negatives_per_frame = max_negatives_per_frame
        self.stage1_balance_ratio = stage1_balance_ratio
        self.stage2_balance_strategy = stage2_balance_strategy
        self.use_group_sampling = use_group_sampling
        self.max_group_samples_ratio = max_group_samples_ratio
        
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
        
        # Apply sample balancing
        if split == 'train':  # Only balance training set
            self.samples = self._balance_samples()
        
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
    
    def _is_within_distance(self, bbox1, bbox2, image_width, distance_multiplier):
        """
        Check if two bounding boxes are within distance limit
        Considers panoramic image wrapping (leftmost and rightmost are connected)
        """
        # Get center points
        center1_x = bbox1[0] + bbox1[2] / 2
        center2_x = bbox2[0] + bbox2[2] / 2
        
        # Calculate x-direction distance
        direct_distance = abs(center1_x - center2_x)
        wrap_distance = image_width - direct_distance if image_width else float('inf')
        actual_distance = min(direct_distance, wrap_distance)
        
        # Distance threshold based on the first person's box width
        person1_width = bbox1[2]
        max_distance = person1_width * distance_multiplier
        
        return actual_distance <= max_distance
    
    def _find_interaction_groups(self, frame_annotations):
        """
        Find interaction groups using graph-based clustering
        If A interacts with B, and B interacts with C, then A, B, C form a group
        """
        # Build interaction graph
        interaction_pairs = set()
        person_ids = set()
        
        for person_data in frame_annotations:
            if 'H-interaction' not in person_data:
                continue
                
            person_id = person_data['label_id']
            person_ids.add(person_id)
            
            for interaction in person_data['H-interaction']:
                if 'pair' in interaction:
                    pair_id = interaction['pair']
                    person_ids.add(pair_id)
                    # Add both directions for undirected graph
                    interaction_pairs.add((person_id, pair_id))
                    interaction_pairs.add((pair_id, person_id))
        
        # Find connected components (groups)
        groups = []
        unvisited = set(person_ids)
        
        while unvisited:
            # Start BFS from an unvisited person
            start_person = unvisited.pop()
            current_group = {start_person}
            queue = [start_person]
            
            while queue:
                current_person = queue.pop(0)
                
                # Find all people this person interacts with
                for person1, person2 in interaction_pairs:
                    if person1 == current_person and person2 in unvisited:
                        current_group.add(person2)
                        queue.append(person2)
                        unvisited.remove(person2)
            
            if len(current_group) > 1:  # Only groups with 2+ people
                groups.append(current_group)
        
        return groups
    
    def _sample_from_group(self, group_members, pedestrians, interaction_data, max_samples):
        """
        Sample interaction pairs from a group using group inference principle
        For a group of n people, sample at most max_samples pairs
        """
        group_list = list(group_members)
        group_size = len(group_list)
        
        if group_size < 2:
            return []
        
        # Calculate maximum samples for this group
        max_possible_pairs = group_size * (group_size - 1) // 2  # C(n,2)
        target_samples = min(max_samples, max_possible_pairs)
        
        # Collect all actual interaction pairs in this group with their interaction types
        actual_pairs = {}
        for person_id in group_list:
            if person_id not in pedestrians:
                continue
            person_data = pedestrians[person_id]
            if 'H-interaction' not in person_data:
                continue
                
            for interaction in person_data['H-interaction']:
                if 'pair' not in interaction or 'inter_labels' not in interaction:
                    continue
                    
                pair_id = interaction['pair']
                if pair_id in group_members:
                    # Canonical pair representation (sorted)
                    pair_key = tuple(sorted([person_id, pair_id]))
                    interaction_type = list(interaction['inter_labels'].keys())[0]
                    actual_pairs[pair_key] = interaction_type
        
        # If we have enough actual pairs, sample from them
        if len(actual_pairs) >= target_samples:
            sampled_pairs = random.sample(list(actual_pairs.items()), target_samples)
            return [(pair[0], pair[1], interaction_type) for pair, interaction_type in sampled_pairs]
        
        # Otherwise, use group inference to fill remaining samples
        sampled_pairs = list(actual_pairs.items())
        remaining_samples = target_samples - len(sampled_pairs)
        
        if remaining_samples > 0:
            # Generate all possible pairs in the group
            all_possible_pairs = []
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    pair_key = tuple(sorted([group_list[i], group_list[j]]))
                    if pair_key not in actual_pairs:  # Only pairs without explicit interaction
                        all_possible_pairs.append(pair_key)
            
            # Sample remaining pairs and infer interaction type
            if all_possible_pairs and remaining_samples > 0:
                # Infer interaction type from the most common type in the group
                interaction_types = list(actual_pairs.values())
                if interaction_types:
                    most_common_type = Counter(interaction_types).most_common(1)[0][0]
                else:
                    most_common_type = 'walking together'  # Default fallback
                
                # Sample remaining pairs
                sample_size = min(remaining_samples, len(all_possible_pairs))
                inferred_pairs = random.sample(all_possible_pairs, sample_size)
                
                for pair_key in inferred_pairs:
                    sampled_pairs.append((pair_key, most_common_type))
        
        # Convert to final format
        result = []
        for pair, interaction_type in sampled_pairs:
            result.append((pair[0], pair[1], interaction_type))
        
        return result
    
    def _load_dataset(self):
        """Load and process the dataset with distance-based pairing"""
        samples = []
        selected_scenes = self._get_scene_splits()
        
        if len(selected_scenes) == 0:
            print("No scenes found!")
            return []
        
        interaction_counts = Counter()
        total_positive_pairs = 0
        total_negative_pairs = 0
        
        for scene_name in selected_scenes:
            # Load social annotations
            social_file = os.path.join(self.social_labels_dir, f"{scene_name}.json")
            if not os.path.exists(social_file):
                continue
            
            try:
                with open(social_file, 'r') as f:
                    social_data = json.load(f)
            except Exception as e:
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
                        pass
            
            # Process frames in this scene
            scene_samples, scene_pos, scene_neg = self._process_scene(
                scene_name, social_data, pose_data, interaction_counts
            )
            samples.extend(scene_samples)
            total_positive_pairs += scene_pos
            total_negative_pairs += scene_neg
        
        print(f"Processed {len(selected_scenes)} scenes: {total_positive_pairs} positive, {total_negative_pairs} negative pairs (ratio 1:{total_negative_pairs/max(total_positive_pairs, 1):.1f})")
        
        # Filter and map interactions to top-4 + others
        filtered_samples = self._filter_and_map_interactions(samples, interaction_counts)
        
        return filtered_samples
    
    def _process_scene(self, scene_name, social_data, pose_data, interaction_counts):
        """Process all frames in a scene with distance-limited pairing"""
        samples = []
        scene_positive_pairs = 0
        scene_negative_pairs = 0
        
        # Access the labels dictionary in social data
        if 'labels' not in social_data:
            return samples, 0, 0
        
        labels_data = social_data['labels']
        
        # Get approximate image width for distance calculation
        # Try to get from first available image
        image_width = None
        for frame_name in list(labels_data.keys())[:5]:  # Check first few frames
            image_path = os.path.join(self.images_dir, scene_name, frame_name)
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        image_width = img.width
                        break
                except:
                    continue
        
        if image_width is None:
            image_width = 1920  # Default assumption for JRDB panoramic images
        
        for frame_name, frame_annotations in labels_data.items():
            # Build full image path
            image_path = os.path.join(self.images_dir, scene_name, frame_name)
            
            if not os.path.exists(image_path):
                continue
            
            # Extract interaction samples from this frame
            frame_samples, frame_pos, frame_neg = self._extract_interactions_from_frame(
                image_path, frame_annotations, pose_data, frame_name, 
                interaction_counts, image_width
            )
            samples.extend(frame_samples)
            scene_positive_pairs += frame_pos
            scene_negative_pairs += frame_neg
        
        return samples, scene_positive_pairs, scene_negative_pairs
    
    def _extract_interactions_from_frame(self, image_path, frame_annotations, pose_data, 
                                       frame_name, interaction_counts, image_width):
        """Extract interaction samples from a single frame using group-based sampling"""
        samples = []
        frame_positive_pairs = 0
        frame_negative_pairs = 0
        
        # Build pedestrian dictionary for quick lookup
        pedestrians = {}
        for person_data in frame_annotations:
            if 'label_id' in person_data and 'box' in person_data:
                ped_id = person_data['label_id']
                pedestrians[ped_id] = person_data
        
        if len(pedestrians) == 0:
            return samples, frame_positive_pairs, frame_negative_pairs
        
        # Use group-based sampling for positive samples
        if self.use_group_sampling:
            positive_pairs, sampled_interaction_pairs = self._extract_positive_samples_group_based(
                frame_annotations, pedestrians, image_width, interaction_counts
            )
        else:
            positive_pairs, sampled_interaction_pairs = self._extract_positive_samples_original(
                frame_annotations, pedestrians, image_width, interaction_counts
            )
        
        # Create positive samples
        for person1_id, person2_id, interaction_type in positive_pairs:
            if person1_id not in pedestrians or person2_id not in pedestrians:
                continue
            
            person1_box = pedestrians[person1_id]['box']
            person2_box = pedestrians[person2_id]['box']
            
            sample = {
                'image_path': image_path,
                'bbox1': person1_box,
                'bbox2': person2_box,
                'interaction': interaction_type,
                'has_interaction': 1,
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
            frame_positive_pairs += 1
        
        # Generate negative samples with distance constraints
        all_person_ids = list(pedestrians.keys())
        target_negative_count = min(
            self.max_negatives_per_frame,
            int(frame_positive_pairs * self.stage1_balance_ratio)
        )
        
        negative_candidates = []
        
        # Find all valid negative pairs within distance
        for i, person1_id in enumerate(all_person_ids):
            person1_box = pedestrians[person1_id]['box']
            
            for j, person2_id in enumerate(all_person_ids[i+1:], start=i+1):
                person2_box = pedestrians[person2_id]['box']
                
                # Skip if already interacting (from sampled pairs)
                pair_tuple = tuple(sorted([person1_id, person2_id]))
                if pair_tuple in sampled_interaction_pairs:
                    continue
                
                # Check distance constraint
                if not self._is_within_distance(person1_box, person2_box, image_width, self.distance_multiplier):
                    continue
                
                negative_candidates.append((person1_id, person2_id, person1_box, person2_box))
        
        # Randomly sample negative pairs
        if len(negative_candidates) > target_negative_count:
            sampled_negatives = random.sample(negative_candidates, target_negative_count)
        else:
            sampled_negatives = negative_candidates
        
        # Create negative samples
        for person1_id, person2_id, person1_box, person2_box in sampled_negatives:
            sample = {
                'image_path': image_path,
                'bbox1': person1_box,
                'bbox2': person2_box,
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
            frame_negative_pairs += 1
        
        return samples, frame_positive_pairs, frame_negative_pairs
    
    def _extract_positive_samples_group_based(self, frame_annotations, pedestrians, image_width, interaction_counts):
        """Extract positive samples using group-based sampling with inference"""
        positive_pairs = []
        sampled_interaction_pairs = set()
        
        # Find interaction groups
        groups = self._find_interaction_groups(frame_annotations)
        
        # Sample from each group
        for group in groups:
            group_size = len(group)
            max_samples_for_group = max(1, int(group_size * self.max_group_samples_ratio))
            
            # Sample pairs from this group
            group_samples = self._sample_from_group(group, pedestrians, frame_annotations, max_samples_for_group)
            
            # Filter by distance constraint
            valid_group_samples = []
            for person1_id, person2_id, interaction_type in group_samples:
                if person1_id not in pedestrians or person2_id not in pedestrians:
                    continue
                    
                person1_box = pedestrians[person1_id]['box']
                person2_box = pedestrians[person2_id]['box']
                
                if self._is_within_distance(person1_box, person2_box, image_width, self.distance_multiplier):
                    valid_group_samples.append((person1_id, person2_id, interaction_type))
                    sampled_interaction_pairs.add(tuple(sorted([person1_id, person2_id])))
                    interaction_counts[interaction_type] += 1
            
            positive_pairs.extend(valid_group_samples)
        
        return positive_pairs, sampled_interaction_pairs
    
    def _extract_positive_samples_original(self, frame_annotations, pedestrians, image_width, interaction_counts):
        """Original positive sample extraction method (for comparison)"""
        positive_pairs = []
        sampled_interaction_pairs = set()
        
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
                
                # Check distance constraint
                if not self._is_within_distance(person_box, pair_box, image_width, self.distance_multiplier):
                    continue
                
                # Avoid duplicate pairs
                pair_tuple = tuple(sorted([person_id, pair_id]))
                if pair_tuple in sampled_interaction_pairs:
                    continue
                
                positive_pairs.append((person_id, pair_id, interaction_type))
                sampled_interaction_pairs.add(pair_tuple)
                interaction_counts[interaction_type] += 1
        
        return positive_pairs, sampled_interaction_pairs
    
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
        
        
        # Update interaction labels with actual found interactions
        self.actual_top_interactions = top_interaction_types
        self.actual_interaction_labels = top_interaction_types + ['others']
        
        return filtered_samples
    
    def _balance_samples(self):
        """Apply sample balancing strategies for training"""
        
        # Separate samples by stages
        stage1_samples = {'positive': [], 'negative': []}
        stage2_samples = {i: [] for i in range(5)}  # 5 interaction classes
        
        for sample in self.samples:
            # Stage 1 classification
            if sample['has_interaction'] == 1:
                stage1_samples['positive'].append(sample)
                # Stage 2 classification (only for positive samples)
                if 'interaction_label' in sample:
                    stage2_samples[sample['interaction_label']].append(sample)
            else:
                stage1_samples['negative'].append(sample)
        
        
        # Balance Stage 1 samples
        balanced_samples = self._balance_stage1_samples(stage1_samples)
        
        # Balance Stage 2 samples if strategy is not None
        if self.stage2_balance_strategy != 'none':
            balanced_samples = self._balance_stage2_samples(balanced_samples, stage2_samples)
        
        return balanced_samples
    
    def _balance_stage1_samples(self, stage1_samples):
        """Balance positive and negative samples for stage 1"""
        pos_count = len(stage1_samples['positive'])
        neg_count = len(stage1_samples['negative'])
        
        target_neg_count = int(pos_count * self.stage1_balance_ratio)
        
        if neg_count > target_neg_count:
            # Downsample negative samples
            sampled_negatives = random.sample(stage1_samples['negative'], target_neg_count)
        else:
            # Keep all negatives (may need to generate more, but we'll keep it simple)
            sampled_negatives = stage1_samples['negative']
        
        balanced_samples = stage1_samples['positive'] + sampled_negatives
        random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def _balance_stage2_samples(self, samples, stage2_samples):
        """Balance interaction type samples for stage 2"""
        if self.stage2_balance_strategy == 'oversample':
            return self._oversample_stage2(samples, stage2_samples)
        elif self.stage2_balance_strategy == 'undersample':
            return self._undersample_stage2(samples, stage2_samples)
        else:
            return samples
    
    def _oversample_stage2(self, samples, stage2_samples):
        """Oversample minority classes in stage 2"""
        # Find the maximum class count
        class_counts = [(i, len(class_samples)) for i, class_samples in stage2_samples.items()]
        class_counts = [count for i, count in class_counts if count > 0]
        
        if not class_counts:
            return samples
        
        max_count = max(class_counts)
        target_count = max_count // 2  # Don't make it completely balanced to avoid overfitting
        
        # Collect all negative samples (stage 1)
        negative_samples = [s for s in samples if s['has_interaction'] == 0]
        positive_samples = []
        
        # Oversample positive samples
        for class_idx, class_samples in stage2_samples.items():
            if len(class_samples) == 0:
                continue
            
            if len(class_samples) < target_count:
                # Oversample this class
                needed = target_count - len(class_samples)
                oversampled = random.choices(class_samples, k=needed)
                positive_samples.extend(class_samples + oversampled)
            else:
                positive_samples.extend(class_samples)
        
        balanced_samples = positive_samples + negative_samples
        random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def _undersample_stage2(self, samples, stage2_samples):
        """Undersample majority classes in stage 2"""
        # Find the minimum class count (exclude empty classes)
        class_counts = [(i, len(class_samples)) for i, class_samples in stage2_samples.items() if len(class_samples) > 0]
        
        if not class_counts:
            return samples
        
        min_count = min(count for i, count in class_counts)
        target_count = min_count * 2  # Keep it reasonable
        
        # Collect all negative samples (stage 1)
        negative_samples = [s for s in samples if s['has_interaction'] == 0]
        positive_samples = []
        
        # Undersample positive samples
        for class_idx, class_samples in stage2_samples.items():
            if len(class_samples) == 0:
                continue
            
            if len(class_samples) > target_count:
                # Undersample this class
                undersampled = random.sample(class_samples, target_count)
                positive_samples.extend(undersampled)
            else:
                positive_samples.extend(class_samples)
        
        balanced_samples = positive_samples + negative_samples
        random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def _print_statistics(self):
        """Print dataset statistics"""
        if len(self.samples) == 0:
            print("No samples loaded!")
            return
        
        # Count by interaction type
        has_interaction_counts = Counter()
        for sample in self.samples:
            has_interaction_counts[sample['has_interaction']] += 1
        
        positive = has_interaction_counts.get(1, 0)
        negative = has_interaction_counts.get(0, 0)
        
        print(f"{self.split} dataset ready: {len(self.samples)} samples ({positive} positive, {negative} negative)")
    
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


def get_optimized_data_loaders(data_path, batch_size=16, num_workers=4, image_size=(224, 224), 
                              use_pose=False, distance_multiplier=3.0, max_negatives_per_frame=5,
                              stage1_balance_ratio=1.0, stage2_balance_strategy='oversample',
                              use_group_sampling=True, max_group_samples_ratio=1.0):
    """
    Create optimized data loaders for train, validation, and test sets
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
    
    # Create datasets with optimizations
    train_dataset = OptimizedJRDBInteractionDataset(
        data_path, split='train', transform=train_transform, 
        image_size=image_size, use_pose=use_pose,
        distance_multiplier=distance_multiplier,
        max_negatives_per_frame=max_negatives_per_frame,
        stage1_balance_ratio=stage1_balance_ratio,
        stage2_balance_strategy=stage2_balance_strategy,
        use_group_sampling=use_group_sampling,
        max_group_samples_ratio=max_group_samples_ratio
    )
    
    # Val and test datasets don't need balancing but can use group sampling
    val_dataset = OptimizedJRDBInteractionDataset(
        data_path, split='val', transform=val_test_transform, 
        image_size=image_size, use_pose=use_pose,
        distance_multiplier=distance_multiplier,
        max_negatives_per_frame=max_negatives_per_frame,
        stage1_balance_ratio=1.0,  # No balancing for validation
        stage2_balance_strategy='none',
        use_group_sampling=use_group_sampling,
        max_group_samples_ratio=max_group_samples_ratio
    )
    
    test_dataset = OptimizedJRDBInteractionDataset(
        data_path, split='test', transform=val_test_transform, 
        image_size=image_size, use_pose=use_pose,
        distance_multiplier=distance_multiplier,
        max_negatives_per_frame=max_negatives_per_frame,
        stage1_balance_ratio=1.0,  # No balancing for test
        stage2_balance_strategy='none',
        use_group_sampling=use_group_sampling,
        max_group_samples_ratio=max_group_samples_ratio
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
                                ['walking_together', 'standing_together', 'conversation', 'sitting_together', 'others'])
    
    return train_loader, val_loader, test_loader, interaction_labels


if __name__ == '__main__':
    # Test the optimized dataset
    data_path = 'D:/1data/imagedata'
    
    print("Testing optimized JRDB dataset loading...")
    
    try:
        # Test dataset with distance constraints
        dataset = OptimizedJRDBInteractionDataset(
            data_path, split='train', use_pose=False,
            distance_multiplier=3.0, max_negatives_per_frame=3,
            stage1_balance_ratio=1.0, stage2_balance_strategy='oversample'
        )
        
        if len(dataset) > 0:
            print(f"Optimized dataset loaded successfully with {len(dataset)} samples")
            
            # Test a sample
            sample = dataset[0]
            print(f"Sample image shape: {sample['image'].shape}")
            print(f"Stage 1 label: {sample['stage1_label']}")
            print(f"Stage 2 label: {sample['stage2_label']}")
            
            # Test data loaders
            print("\nTesting optimized data loaders...")
            train_loader, val_loader, test_loader, labels = get_optimized_data_loaders(
                data_path, batch_size=4, num_workers=0,  # Use 0 workers for testing
                distance_multiplier=3.0, max_negatives_per_frame=3
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
        print(f"Error testing optimized dataset: {e}")
        import traceback
        traceback.print_exc()