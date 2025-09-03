import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import MyMobileNet


class TwoStageInteractionClassifier(nn.Module):
    """
    Two-stage classification network for human interaction detection
    Stage 1: Binary classification - whether interaction exists
    Stage 2: Multi-class classification - interaction type (top 4 + other)
    """
    
    def __init__(self, backbone_name='mobilenet', pretrained=True, num_interaction_classes=5):
        super(TwoStageInteractionClassifier, self).__init__()
        
        self.num_interaction_classes = num_interaction_classes
        
        # Initialize backbone
        if backbone_name == 'mobilenet':
            self.backbone = MyMobileNet(pretrained=pretrained)
            # MobileNetV2 output channels: 1280
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Stage 1: Binary interaction detection (interaction vs no interaction)
        self.stage1_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary classification
        )
        
        # Stage 2: Interaction type classification (5 classes: top 4 + other)
        self.stage2_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_interaction_classes)
        )
        
    def forward(self, x, stage='both'):
        """
        Forward pass
        Args:
            x: Input images [batch_size, 3, H, W]
            stage: 'stage1', 'stage2', or 'both'
        Returns:
            Dictionary containing stage outputs
        """
        # Extract features using backbone
        features = self.backbone(x)[0]  # Get the last feature map
        
        # Global average pooling
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        results = {}
        
        if stage in ['stage1', 'both']:
            # Stage 1: Binary interaction detection
            stage1_output = self.stage1_classifier(pooled_features)
            results['stage1'] = stage1_output
        
        if stage in ['stage2', 'both']:
            # Stage 2: Interaction type classification
            stage2_output = self.stage2_classifier(pooled_features)
            results['stage2'] = stage2_output
            
        return results
    
    def predict(self, x, threshold=0.5):
        """
        Make predictions with both stages
        Args:
            x: Input images
            threshold: Threshold for stage 1 binary classification
        Returns:
            Final predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, stage='both')
            
            # Stage 1: Check if interaction exists
            stage1_probs = F.softmax(outputs['stage1'], dim=1)
            has_interaction = stage1_probs[:, 1] > threshold  # Probability of interaction
            
            # Stage 2: Get interaction type
            stage2_probs = F.softmax(outputs['stage2'], dim=1)
            interaction_type = torch.argmax(stage2_probs, dim=1)
            
            # Final prediction: if no interaction detected, set type to -1
            final_predictions = torch.where(has_interaction, interaction_type, 
                                          torch.tensor(-1, device=x.device))
            
            return {
                'has_interaction': has_interaction,
                'interaction_type': interaction_type,
                'final_prediction': final_predictions,
                'stage1_probs': stage1_probs,
                'stage2_probs': stage2_probs
            }


class InteractionLoss(nn.Module):
    """
    Combined loss for two-stage training
    """
    
    def __init__(self, stage1_weight=1.0, stage2_weight=1.0, 
                 class_weights=None, focal_alpha=1.0, focal_gamma=2.0):
        super(InteractionLoss, self).__init__()
        
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Stage 1: Binary cross entropy
        self.stage1_criterion = nn.CrossEntropyLoss()
        
        # Stage 2: Weighted cross entropy for class imbalance
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.stage2_criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.stage2_criterion = nn.CrossEntropyLoss()
    
    def focal_loss(self, inputs, targets, alpha=1.0, gamma=2.0):
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, outputs, stage1_targets, stage2_targets, stage='both'):
        """
        Calculate loss for training
        Args:
            outputs: Model outputs dictionary
            stage1_targets: Binary interaction labels
            stage2_targets: Interaction type labels
            stage: Which stage to calculate loss for
        """
        total_loss = 0
        loss_dict = {}
        
        if stage in ['stage1', 'both'] and 'stage1' in outputs:
            stage1_loss = self.stage1_criterion(outputs['stage1'], stage1_targets)
            loss_dict['stage1_loss'] = stage1_loss
            total_loss += self.stage1_weight * stage1_loss
        
        if stage in ['stage2', 'both'] and 'stage2' in outputs:
            # Only calculate stage2 loss for samples with interactions
            interaction_mask = stage1_targets == 1
            
            if interaction_mask.sum() > 0:
                stage2_outputs_filtered = outputs['stage2'][interaction_mask]
                stage2_targets_filtered = stage2_targets[interaction_mask]
                
                # Use focal loss for stage 2 to handle class imbalance
                stage2_loss = self.focal_loss(stage2_outputs_filtered, 
                                            stage2_targets_filtered,
                                            self.focal_alpha, self.focal_gamma)
                loss_dict['stage2_loss'] = stage2_loss
                total_loss += self.stage2_weight * stage2_loss
            else:
                loss_dict['stage2_loss'] = torch.tensor(0.0, device=outputs['stage2'].device)
        
        loss_dict['total_loss'] = total_loss
        return loss_dict


# Interaction type mapping (based on JRDB project plan)
INTERACTION_MAPPING = {
    'walking together': 0,      # 46.9%
    'standing together': 1,     # 33.7%
    'conversation': 2,          # 8.8%
    'sitting together': 3,      # 7.7%
    # All other 15 types -> 4   # 2.9% total
}

# Reverse mapping for interpretation
INTERACTION_LABELS = [
    'walking_together',
    'standing_together', 
    'conversation',
    'sitting_together',
    'others'
]

def map_interaction_label(original_label):
    """Map original H-interaction labels to 5-class system"""
    return INTERACTION_MAPPING.get(original_label, 4)  # Others -> 4

def get_class_weights():
    """Get class weights based on frequency (inverse frequency weighting)"""
    frequencies = [0.469, 0.337, 0.088, 0.077, 0.029]
    weights = [1.0 / freq for freq in frequencies]
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight * len(weights) for w in weights]
    return weights


if __name__ == '__main__':
    # Test the model
    model = TwoStageInteractionClassifier(backbone_name='mobilenet', pretrained=True)
    
    # Dummy input
    x = torch.randn(4, 3, 224, 224)
    
    # Test forward pass
    outputs = model(x, stage='both')
    print("Stage 1 output shape:", outputs['stage1'].shape)
    print("Stage 2 output shape:", outputs['stage2'].shape)
    
    # Test prediction
    predictions = model.predict(x)
    print("Predictions keys:", list(predictions.keys()))
    
    # Test loss function
    class_weights = get_class_weights()
    criterion = InteractionLoss(class_weights=class_weights)
    
    stage1_targets = torch.randint(0, 2, (4,))
    stage2_targets = torch.randint(0, 5, (4,))
    
    loss_dict = criterion(outputs, stage1_targets, stage2_targets)
    print("Loss components:", loss_dict)