import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import MyMobileNet, MyVGG16, MyVGG19, MyRes18, MyRes50, MyAlex, MyInception_v3


class DualPersonInteractionClassifier(nn.Module):
    """
    Improved two-stage classification network with dual-person feature fusion
    
    Architecture:
    Person A Box → Backbone → Feature_A [1280]
                                         ↓
    Person B Box → Backbone → Feature_B [1280] → Fusion → Stage1 & Stage2
    
    Key improvements:
    1. Individual person feature extraction
    2. Multiple fusion strategies
    3. Better handling of crowded scenes
    """
    
    def __init__(self, backbone_name='mobilenet', pretrained=True, num_interaction_classes=5, 
                 fusion_method='concat', shared_backbone=False):
        super(DualPersonInteractionClassifier, self).__init__()
        
        self.num_interaction_classes = num_interaction_classes
        self.fusion_method = fusion_method
        self.shared_backbone = shared_backbone
        
        # Initialize backbone(s)
        if backbone_name == 'mobilenet':
            self.backbone_A = MyMobileNet(pretrained=pretrained)
            if shared_backbone:
                self.backbone_B = self.backbone_A  # Share weights
            else:
                self.backbone_B = MyMobileNet(pretrained=pretrained)  # Separate weights
            self.feature_dim = 1280
        elif backbone_name == 'vgg16':
            self.backbone_A = MyVGG16(pretrained=pretrained)
            if shared_backbone:
                self.backbone_B = self.backbone_A
            else:
                self.backbone_B = MyVGG16(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone_name == 'vgg19':
            self.backbone_A = MyVGG19(pretrained=pretrained)
            if shared_backbone:
                self.backbone_B = self.backbone_A
            else:
                self.backbone_B = MyVGG19(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone_name == 'resnet18':
            self.backbone_A = MyRes18(pretrained=pretrained)
            if shared_backbone:
                self.backbone_B = self.backbone_A
            else:
                self.backbone_B = MyRes18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone_name == 'resnet50':
            self.backbone_A = MyRes50(pretrained=pretrained)
            if shared_backbone:
                self.backbone_B = self.backbone_A
            else:
                self.backbone_B = MyRes50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone_name == 'alexnet':
            self.backbone_A = MyAlex(pretrained=pretrained)
            if shared_backbone:
                self.backbone_B = self.backbone_A
            else:
                self.backbone_B = MyAlex(pretrained=pretrained)
            self.feature_dim = 256
        elif backbone_name == 'inception_v3':
            self.backbone_A = MyInception_v3(pretrained=pretrained)
            if shared_backbone:
                self.backbone_B = self.backbone_A
            else:
                self.backbone_B = MyInception_v3(pretrained=pretrained)
            self.feature_dim = 768  # Using Mixed_6e output
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Global average pooling for both persons
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine fusion feature dimension
        if fusion_method == 'concat':
            self.fused_feature_dim = self.feature_dim * 2  # 2560
        elif fusion_method in ['add', 'subtract', 'multiply']:
            self.fused_feature_dim = self.feature_dim  # 1280
        elif fusion_method == 'attention':
            self.fused_feature_dim = self.feature_dim  # 1280
            # Attention mechanism
            self.attention_net = nn.Sequential(
                nn.Linear(self.feature_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Stage 1: Binary interaction detection
        self.stage1_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fused_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary classification
        )
        
        # Stage 2: Interaction type classification
        self.stage2_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fused_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_interaction_classes)
        )
        
        print(f"Dual-Person Classifier initialized:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Shared backbone: {shared_backbone}")
        print(f"  Fusion method: {fusion_method}")
        print(f"  Feature dimensions: {self.feature_dim} -> {self.fused_feature_dim}")
    
    def extract_person_features(self, person_images, backbone):
        """Extract features for a batch of person images"""
        # Input: [batch_size, 3, H, W]
        feature_maps = backbone(person_images)  # Returns list of feature maps
        
        # For Inception_v3, use the last feature map (Mixed_6e)
        if isinstance(backbone, MyInception_v3):
            features = feature_maps[-1]  # Use Mixed_6e (768 channels)
        else:
            features = feature_maps[0]  # For other networks, use the only output
            
        pooled_features = self.global_pool(features)  # [batch_size, feature_dim, 1, 1]
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # [batch_size, feature_dim]
        return pooled_features
    
    def fuse_features(self, features_A, features_B):
        """Fuse features from two persons using specified method"""
        batch_size = features_A.size(0)
        
        if self.fusion_method == 'concat':
            # Concatenation: [batch_size, feature_dim * 2]
            fused_features = torch.cat([features_A, features_B], dim=1)
            
        elif self.fusion_method == 'add':
            # Element-wise addition: [batch_size, feature_dim]
            fused_features = features_A + features_B
            
        elif self.fusion_method == 'subtract':
            # Element-wise subtraction (captures differences): [batch_size, feature_dim]
            fused_features = torch.abs(features_A - features_B)
            
        elif self.fusion_method == 'multiply':
            # Element-wise multiplication (captures interactions): [batch_size, feature_dim]
            fused_features = features_A * features_B
            
        elif self.fusion_method == 'attention':
            # Attention-weighted combination: [batch_size, feature_dim]
            concat_features = torch.cat([features_A, features_B], dim=1)
            attention_weights = self.attention_net(concat_features)  # [batch_size, 2]
            
            # Apply attention weights
            weighted_A = features_A * attention_weights[:, 0:1]  # [batch_size, feature_dim]
            weighted_B = features_B * attention_weights[:, 1:2]  # [batch_size, feature_dim]
            fused_features = weighted_A + weighted_B
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_features
    
    def forward(self, person_A_images, person_B_images, stage='both'):
        """
        Forward pass with dual person inputs
        
        Args:
            person_A_images: Images of person A [batch_size, 3, H, W]
            person_B_images: Images of person B [batch_size, 3, H, W]
            stage: 'stage1', 'stage2', or 'both'
        
        Returns:
            Dictionary containing stage outputs
        """
        # Extract features for both persons
        features_A = self.extract_person_features(person_A_images, self.backbone_A)
        features_B = self.extract_person_features(person_B_images, self.backbone_B)
        
        # Fuse features
        fused_features = self.fuse_features(features_A, features_B)
        
        results = {
            'features_A': features_A,
            'features_B': features_B,
            'fused_features': fused_features
        }
        
        if stage in ['stage1', 'both']:
            # Stage 1: Binary interaction detection
            stage1_output = self.stage1_classifier(fused_features)
            results['stage1'] = stage1_output
        
        if stage in ['stage2', 'both']:
            # Stage 2: Interaction type classification
            stage2_output = self.stage2_classifier(fused_features)
            results['stage2'] = stage2_output
            
        return results
    
    def predict(self, person_A_images, person_B_images, threshold=0.5):
        """
        Make predictions with both stages
        
        Args:
            person_A_images: Images of person A
            person_B_images: Images of person B
            threshold: Threshold for stage 1 binary classification
        
        Returns:
            Final predictions with individual and fused features
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(person_A_images, person_B_images, stage='both')
            
            # Stage 1: Check if interaction exists
            stage1_probs = F.softmax(outputs['stage1'], dim=1)
            has_interaction = stage1_probs[:, 1] > threshold
            
            # Stage 2: Get interaction type
            stage2_probs = F.softmax(outputs['stage2'], dim=1)
            interaction_type = torch.argmax(stage2_probs, dim=1)
            
            # Final prediction: if no interaction detected, set type to -1
            final_predictions = torch.where(has_interaction, interaction_type, 
                                          torch.tensor(-1, device=person_A_images.device))
            
            return {
                'has_interaction': has_interaction,
                'interaction_type': interaction_type,
                'final_prediction': final_predictions,
                'stage1_probs': stage1_probs,
                'stage2_probs': stage2_probs,
                'features_A': outputs['features_A'],
                'features_B': outputs['features_B'],
                'fused_features': outputs['fused_features']
            }


class DualPersonInteractionLoss(nn.Module):
    """
    Loss function for dual-person interaction classification
    """
    
    def __init__(self, stage1_weight=1.0, stage2_weight=1.0, 
                 class_weights=None, focal_alpha=1.0, focal_gamma=2.0,
                 feature_regularization=False, reg_weight=0.01):
        super(DualPersonInteractionLoss, self).__init__()
        
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.feature_regularization = feature_regularization
        self.reg_weight = reg_weight
        
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
    
    def feature_regularization_loss(self, features_A, features_B):
        """Encourage feature diversity between two persons"""
        # L2 distance between features (encourage diversity)
        similarity = F.cosine_similarity(features_A, features_B, dim=1)
        # Penalize high similarity (encourage different features for different persons)
        reg_loss = torch.mean(similarity ** 2)
        return reg_loss
    
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
        
        # Feature regularization loss
        if self.feature_regularization and 'features_A' in outputs and 'features_B' in outputs:
            reg_loss = self.feature_regularization_loss(outputs['features_A'], outputs['features_B'])
            loss_dict['regularization_loss'] = reg_loss
            total_loss += self.reg_weight * reg_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict


# Fusion method comparison helper
FUSION_METHODS = {
    'concat': 'Concatenation - combines all features',
    'add': 'Addition - symmetric combination',
    'subtract': 'Subtraction - captures differences',
    'multiply': 'Multiplication - captures interactions',
    'attention': 'Attention - learned weighted combination'
}

def get_fusion_method_info():
    """Get information about available fusion methods"""
    return FUSION_METHODS


if __name__ == '__main__':
    # Test the dual-person model
    print("Testing Dual-Person Interaction Classifier...")
    
    # Test different fusion methods
    fusion_methods = ['concat', 'add', 'attention']
    
    for fusion_method in fusion_methods:
        print(f"\n--- Testing fusion method: {fusion_method} ---")
        
        model = DualPersonInteractionClassifier(
            backbone_name='mobilenet', 
            pretrained=True,
            fusion_method=fusion_method,
            shared_backbone=True
        )
        
        # Dummy inputs
        person_A_images = torch.randn(4, 3, 224, 224)
        person_B_images = torch.randn(4, 3, 224, 224)
        
        # Test forward pass
        outputs = model(person_A_images, person_B_images, stage='both')
        print(f"  Stage 1 output shape: {outputs['stage1'].shape}")
        print(f"  Stage 2 output shape: {outputs['stage2'].shape}")
        print(f"  Fused features shape: {outputs['fused_features'].shape}")
        
        # Test prediction
        predictions = model.predict(person_A_images, person_B_images)
        print(f"  Prediction keys: {list(predictions.keys())}")
        
        # Test loss function
        from two_stage_classifier import get_class_weights
        class_weights = get_class_weights()
        criterion = DualPersonInteractionLoss(
            class_weights=class_weights,
            feature_regularization=True
        )
        
        stage1_targets = torch.randint(0, 2, (4,))
        stage2_targets = torch.randint(0, 5, (4,))
        
        loss_dict = criterion(outputs, stage1_targets, stage2_targets)
        print(f"  Loss components: {list(loss_dict.keys())}")
        
        print(f"  ✓ {fusion_method} method working correctly")
    
    print(f"\nDual-Person Classifier implementation completed!")