#!/usr/bin/env python3
"""
Stage2 geometric behavior classifier
Specialized architecture for 5-class behavior classification using 16D geometric+temporal features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class GeometricStage2Classifier(nn.Module):
    """
    专门为5类行为分类设计的几何+时序特征分类器
    输入：16维特征 (7几何 + 4基础运动 + 5增强时序)
    输出：5类行为分类
    """
    
    def __init__(self, input_dim=16, hidden_dims=[64, 32, 16], dropout=0.2, use_attention=True):
        super().__init__()
        
        # 特征维度分组
        self.geometric_dim = 7      # 基础几何特征
        self.basic_motion_dim = 4   # Stage1的运动特征  
        self.enhanced_motion_dim = 5 # 新增时序特征（压缩到5维）
        self.total_dim = self.geometric_dim + self.basic_motion_dim + self.enhanced_motion_dim
        
        assert self.total_dim == input_dim, f"Feature dimensions mismatch: {self.total_dim} != {input_dim}"
        
        # 分组特征处理器
        self.geometric_processor = nn.Sequential(
            nn.Linear(self.geometric_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.basic_motion_processor = nn.Sequential(
            nn.Linear(self.basic_motion_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.enhanced_motion_processor = nn.Sequential(
            nn.Linear(self.enhanced_motion_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征融合层
        fusion_dim = 16 + 8 + 16  # 40
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征注意力机制 (可选)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[0], num_heads=4, batch_first=True
            )
        
        # 行为分类器
        layers = []
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 5))  # 5类分类
        self.classifier = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, features):
        """
        Args:
            features: [batch_size, 16] 完整特征向量
        Returns:
            [batch_size, 5] 分类logits
        """
        batch_size = features.size(0)
        
        # 分组处理特征
        geometric_feat = features[:, :self.geometric_dim]
        basic_motion_feat = features[:, self.geometric_dim:self.geometric_dim+self.basic_motion_dim]
        enhanced_motion_feat = features[:, -self.enhanced_motion_dim:]
        
        # 各组特征处理
        geo_processed = self.geometric_processor(geometric_feat)       # [batch, 16]
        basic_processed = self.basic_motion_processor(basic_motion_feat)  # [batch, 8]
        enhanced_processed = self.enhanced_motion_processor(enhanced_motion_feat)  # [batch, 16]
        
        # 特征融合
        fused_features = torch.cat([geo_processed, basic_processed, enhanced_processed], dim=1)  # [batch, 40]
        
        # 融合层处理
        fusion_output = self.fusion_layer(fused_features)  # [batch, hidden_dims[0]]
        
        # 可选的注意力机制
        if self.use_attention:
            # 重塑为序列格式进行注意力计算
            fused_seq = fusion_output.unsqueeze(1)  # [batch, 1, hidden_dims[0]]
            attended_features, _ = self.attention(fused_seq, fused_seq, fused_seq)
            fusion_output = attended_features.squeeze(1)  # [batch, hidden_dims[0]]
        
        # 最终分类
        logits = self.classifier(fusion_output)  # [batch, 5]
        
        return logits
    
    def get_feature_importance(self, features):
        """
        分析特征重要性（通过梯度计算）
        
        Args:
            features: [batch_size, 16] 输入特征
        Returns:
            Dict: 各组特征的重要性分数
        """
        self.eval()
        features.requires_grad_(True)
        
        # 前向传播
        logits = self.forward(features)
        
        # 计算各类的梯度
        importance_scores = {}
        for class_id in range(5):
            # 计算对该类的梯度
            class_score = logits[:, class_id].sum()
            grads = torch.autograd.grad(class_score, features, retain_graph=True)[0]
            
            # 计算各特征组的重要性
            geo_importance = torch.mean(torch.abs(grads[:, :self.geometric_dim]))
            basic_importance = torch.mean(torch.abs(grads[:, self.geometric_dim:self.geometric_dim+self.basic_motion_dim]))
            enhanced_importance = torch.mean(torch.abs(grads[:, -self.enhanced_motion_dim:]))
            
            importance_scores[f'class_{class_id}'] = {
                'geometric': geo_importance.item(),
                'basic_motion': basic_importance.item(),
                'enhanced_motion': enhanced_importance.item()
            }
        
        return importance_scores


class Stage2Loss(nn.Module):
    """Stage2专用损失函数：加权交叉熵 + MPCA + 准确率正则"""
    
    def __init__(self, class_weights, mpca_weight=0.1, acc_weight=0.05):
        super().__init__()
        if isinstance(class_weights, dict):
            self.class_weights = torch.tensor(list(class_weights.values()), dtype=torch.float32)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.mpca_weight = mpca_weight
        self.acc_weight = acc_weight
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, 5] 预测logits
            targets: [batch_size] 真实标签
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        # 主损失：加权交叉熵
        ce_loss = F.cross_entropy(
            predictions, targets, 
            weight=self.class_weights.to(predictions.device)
        )
        
        # MPCA正则化：最小化类别间准确率方差
        with torch.no_grad():
            pred_classes = torch.argmax(predictions, dim=1)
            per_class_acc = []
            
            for class_id in range(5):
                mask = targets == class_id
                if mask.sum() > 0:
                    class_acc = (pred_classes[mask] == targets[mask]).float().mean()
                    per_class_acc.append(class_acc)
            
            if len(per_class_acc) > 1:
                mpca_loss = torch.std(torch.stack(per_class_acc))
            else:
                mpca_loss = torch.tensor(0.0, device=predictions.device)
        
        # 准确率正则化：鼓励整体准确率提升
        overall_acc = (pred_classes == targets).float().mean()
        acc_regularization = -overall_acc  # 负号表示最大化准确率
        
        # 计算总损失
        total_loss = ce_loss + self.mpca_weight * mpca_loss + self.acc_weight * acc_regularization
        
        # 返回详细信息
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'mpca_loss': mpca_loss.item(),
            'overall_acc': overall_acc.item()
        }
        
        return total_loss, loss_dict


class Stage2Evaluator:
    """Stage2专用评估器"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """重置评估状态"""
        self.predictions = []
        self.targets = []
        self.correct_per_class = {i: 0 for i in range(len(self.class_names))}
        self.total_per_class = {i: 0 for i in range(len(self.class_names))}
    
    def update(self, predictions, targets):
        """
        更新评估结果
        
        Args:
            predictions: [batch_size, 5] logits或[batch_size] 预测类别
            targets: [batch_size] 真实标签
        """
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
        
        # 转换为numpy
        pred_classes = pred_classes.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # 存储预测结果
        self.predictions.extend(pred_classes)
        self.targets.extend(targets)
        
        # 更新各类别统计
        for pred, target in zip(pred_classes, targets):
            self.total_per_class[target] += 1
            if pred == target:
                self.correct_per_class[target] += 1
    
    def compute_metrics(self):
        """计算评估指标"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # 整体准确率
        overall_acc = np.mean(predictions == targets)
        
        # 各类别准确率
        per_class_acc = {}
        valid_classes = []
        
        for class_id in range(len(self.class_names)):
            if self.total_per_class[class_id] > 0:
                acc = self.correct_per_class[class_id] / self.total_per_class[class_id]
                per_class_acc[class_id] = acc
                valid_classes.append(acc)
        
        # MPCA (Mean Per-Class Accuracy)
        mpca = np.mean(valid_classes) if valid_classes else 0.0
        
        # 计算混淆矩阵相关指标
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # 加权平均
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=support)
        
        # 混淆矩阵
        cm = confusion_matrix(targets, predictions, labels=range(len(self.class_names)))
        
        return {
            'overall_accuracy': overall_acc,
            'mpca': mpca,
            'per_class_accuracy': per_class_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'support': support,
            'confusion_matrix': cm,
            'class_names': self.class_names
        }
    
    def print_evaluation_report(self):
        """打印详细的评估报告"""
        metrics = self.compute_metrics()
        
        if not metrics:
            print("No evaluation data available.")
            return
        
        print("\n" + "="*60)
        print("STAGE2 BEHAVIOR CLASSIFICATION EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"MPCA (Mean Per-Class Accuracy): {metrics['mpca']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 80)
        
        for i, class_name in enumerate(metrics['class_names']):
            if i < len(metrics['precision']):
                print(f"{class_name:<25} {metrics['precision'][i]:<10.4f} "
                      f"{metrics['recall'][i]:<10.4f} {metrics['f1_score'][i]:<10.4f} "
                      f"{metrics['support'][i]:<10}")
        
        print("\nPer-Class Accuracy:")
        print("-" * 40)
        for class_id, acc in metrics['per_class_accuracy'].items():
            class_name = metrics['class_names'][class_id]
            print(f"{class_name:<25}: {acc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])


if __name__ == '__main__':
    # 测试Stage2分类器
    print("Testing GeometricStage2Classifier...")
    
    # 创建模型
    model = GeometricStage2Classifier(
        input_dim=16,
        hidden_dims=[64, 32, 16],
        dropout=0.2,
        use_attention=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 8
    features = torch.randn(batch_size, 16)
    targets = torch.randint(0, 5, (batch_size,))
    
    print(f"\nInput shape: {features.shape}")
    
    # 前向传播
    logits = model(features)
    print(f"Output shape: {logits.shape}")
    
    # 测试损失函数
    class_weights = {0: 1.0, 1: 1.4, 2: 8.3, 3: 7.4, 4: 50.0}
    criterion = Stage2Loss(class_weights, mpca_weight=0.1, acc_weight=0.05)
    
    loss, loss_dict = criterion(logits, targets)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss details: {loss_dict}")
    
    # 测试特征重要性
    print(f"\nTesting feature importance...")
    importance = model.get_feature_importance(features)
    print(f"Feature importance (sample): {importance['class_0']}")
    
    # 测试评估器
    print(f"\nTesting evaluator...")
    class_names = ['Static Group', 'Parallel Movement', 'Approaching Interaction', 
                   'Coordinated Activity', 'Complex/Rare Behaviors']
    
    evaluator = Stage2Evaluator(class_names)
    evaluator.update(logits, targets)
    evaluator.print_evaluation_report()
    
    print("\nStage2 classifier test completed!")