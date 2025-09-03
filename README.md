# Two-Stage Human Interaction Classification Network

A PyTorch implementation of a two-stage classification network for human interaction detection and classification.

## Overview

This project implements a two-stage approach for human interaction analysis:

1. **Stage 1**: Binary classification to detect whether two people are interacting or not
2. **Stage 2**: Multi-class classification to determine the type of interaction (top 4 categories + others)

The network uses MobileNetV2 as the backbone feature extractor for efficient processing.

## Key Features

- **Two-stage architecture**: Hierarchical classification approach
- **Class imbalance handling**: Focal loss and weighted cross-entropy
- **Flexible backbone**: Easy to switch between different CNN architectures
- **Comprehensive evaluation**: Detailed metrics and visualizations
- **JRDB dataset support**: Designed for the JRDB social interaction dataset

## Interaction Categories

Based on the JRDB dataset analysis, the top 4 interaction types are:

1. **Walking together** (46.9% of interactions)
2. **Standing together** (33.7% of interactions)
3. **Conversation** (8.8% of interactions)
4. **Sitting together** (7.7% of interactions)
5. **Others** (2.9% - includes 15 other interaction types)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data_path /path/to/dataset --epochs 100 --batch_size 16
```

Key training arguments:
- `--data_path`: Path to your dataset directory
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--backbone`: Backbone network (default: mobilenet)
- `--use_class_weights`: Use class weights for imbalanced data (default: True)

### Evaluation

```bash
python evaluate.py --checkpoint_path ./checkpoints/best_model.pth --data_path /path/to/dataset
```

### Dataset Structure

Expected dataset structure (JRDB format):
```
D:/1data/imagedata/
├── images/image_stitched/              # 图像数据
│   ├── scene1/                         # 场景文件夹  
│   │   ├── 000000.jpg                  # 帧图像
│   │   ├── 000001.jpg
│   │   └── ...
│   └── scene2/
└── labels/
    ├── labels_2d_activity_social_stitched/    # 社交活动标注
    │   ├── scene1.json
    │   └── scene2.json
    └── labels_2d_pose_stitched_coco/          # 姿态标注 (可选)
        ├── scene1.json
        └── scene2.json
```

### JRDB Data Format

The dataset follows the exact JRDB format with:

**Social Annotation Structure:**
```json
{
    "labels": {
        "000000.jpg": [
            {
                "label_id": "pedestrian:6",
                "box": [x, y, width, height],
                "action_label": {"sitting": 1},
                "H-interaction": [
                    {
                        "pair": "pedestrian:7",
                        "box_pair": [x, y, width, height],
                        "inter_labels": {"sitting together": 1}
                    }
                ]
            }
        ]
    }
}
```

**Key Features:**
- Extracts H-interactions between pedestrian pairs
- Maps person IDs: `pedestrian:X` ↔ `track_id` (from pose data)
- Automatically identifies top-4 interaction types from your dataset
- Generates negative samples for no-interaction cases
- Optional pose integration using COCO 17-keypoint format

## Architecture Details

### Model Architecture

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Feature Extraction**: Global average pooling of backbone features
- **Stage 1 Classifier**: FC layers → Binary classification (interaction/no interaction)
- **Stage 2 Classifier**: FC layers → 5-class classification (interaction types)

### Loss Function

- **Stage 1**: Cross-entropy loss for binary classification
- **Stage 2**: Focal loss with class weights to handle imbalanced data
- **Combined**: Weighted sum of stage 1 and stage 2 losses

### Training Strategy

1. **Data Preprocessing**: 
   - Crop regions containing both pedestrians
   - Resize to 224×224
   - Normalize with ImageNet statistics

2. **Data Augmentation**:
   - Random horizontal flip
   - Color jittering
   - Standard normalization

3. **Optimization**:
   - Adam optimizer with weight decay
   - StepLR or Cosine Annealing scheduler
   - Class weights for handling imbalanced data

## File Structure

```
classificationnet/
├── backbone.py              # CNN backbone models (MobileNet, ResNet, etc.)
├── two_stage_classifier.py  # Main model architecture and loss functions
├── dataset.py              # Dataset loading and preprocessing
├── train.py                # Training script with full pipeline
├── evaluate.py             # Evaluation and testing script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Performance Metrics

The model is evaluated using:

- **Stage 1**: Accuracy, Precision, Recall, F1-Score for interaction detection
- **Stage 2**: Multi-class accuracy, Per-class precision/recall for interaction types
- **Visualizations**: Confusion matrices, probability distributions, training curves

## Results

Training generates:
- Model checkpoints (best loss, best accuracy, final)
- Training curves and metrics plots
- Detailed evaluation reports
- Confusion matrices for both stages

## Customization

### Adding New Backbones

To add a new backbone network:

1. Add your backbone class to `backbone.py`
2. Update the backbone selection in `two_stage_classifier.py`
3. Adjust the feature dimension accordingly

### Modifying Loss Functions

The loss function components can be customized in `two_stage_classifier.py`:
- Stage weights (`stage1_weight`, `stage2_weight`)
- Focal loss parameters (`focal_alpha`, `focal_gamma`)
- Class weights for imbalanced data

## Notes

- The model expects RGB images and will crop regions containing both pedestrians
- Negative samples (no interaction) are automatically generated during dataset loading
- Class imbalance is handled through focal loss and weighted sampling
- The model can be easily extended to different numbers of interaction classes

## Citation

This implementation is designed for human interaction analysis research. If you use this code, please consider citing relevant papers on human interaction detection and the JRDB dataset.