#!/usr/bin/env python3
"""
Debug script to check JRDB data loading
"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict

def debug_jrdb_data(data_path: str):
    """调试JRDB数据加载"""
    social_labels_path = Path(data_path) / "labels" / "labels_2d_activity_social_stitched"
    
    if not social_labels_path.exists():
        print(f"❌ Path not found: {social_labels_path}")
        return
    
    json_files = list(social_labels_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # 只处理第一个文件
    json_file = json_files[0]
    print(f"Processing: {json_file.name}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)
    
    labels = scene_data.get('labels', {})
    
    interaction_count = 0
    label_counts = Counter()
    multi_label_examples = []
    
    for frame_name, frame_data in list(labels.items())[:2]:  # 只看前2帧
        print(f"\nFrame: {frame_name}")
        
        for person_idx, person_data in enumerate(frame_data):
            h_interactions = person_data.get('H-interaction', [])
            person_id = person_data.get('label_id', '')
            
            print(f"  Person {person_id}: {len(h_interactions)} interactions")
            
            for interaction_idx, interaction in enumerate(h_interactions):
                inter_labels = interaction.get('inter_labels', {})
                pair_id = interaction.get('pair', '')
                
                # 获取激活的标签
                active_labels = [label for label, value in inter_labels.items() if value > 0]
                
                print(f"    Interaction {interaction_idx} with {pair_id}: {active_labels}")
                
                if active_labels:
                    interaction_count += 1
                    for label in active_labels:
                        label_counts[label] += 1
                    
                    if len(active_labels) > 1:
                        multi_label_examples.append({
                            'frame': frame_name,
                            'person': person_id,
                            'pair': pair_id,
                            'labels': active_labels
                        })
    
    print(f"\n=== Summary ===")
    print(f"Total interactions found: {interaction_count}")
    print(f"Label distribution:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count}")
    
    print(f"\nMulti-label examples ({len(multi_label_examples)}):")
    for example in multi_label_examples[:5]:
        print(f"  {example}")
    
    # 计算共现
    if multi_label_examples:
        print(f"\nCo-occurrence examples:")
        for example in multi_label_examples[:3]:
            labels = example['labels']
            print(f"  {labels[0]} co-occurs with: {labels[1:]}")

if __name__ == '__main__':
    debug_jrdb_data('../dataset')