#!/usr/bin/env python3
"""
JRDB Dataset Label Co-occurrence Analysis
Analyzes label interaction patterns and validates multi-level classification design
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Try to import chord visualization
try:
    from chord import Chord
    CHORD_AVAILABLE = True
except ImportError:
    print("Warning: chord package not available. Install with: pip install chord")
    CHORD_AVAILABLE = False

# For network visualization
try:
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    NETWORK_AVAILABLE = True
except ImportError:
    print("Warning: networkx/plotly not available. Some visualizations will be skipped")
    NETWORK_AVAILABLE = False


class JRDBLabelAnalyzer:
    """JRDB数据集标签共现分析器"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: JRDB数据集路径，包含labels/labels_2d_activity_social_stitched/
        """
        self.data_path = Path(data_path)
        self.social_labels_path = self.data_path / "labels" / "labels_2d_activity_social_stitched"
        
        # 主要关注的三个类别
        self.main_categories = {
            'walking_together': 'walking together',
            'standing_together': 'standing together', 
            'sitting_together': 'sitting together'
        }
        
        # 所有交互标签
        self.all_interactions = set()
        
        # 数据存储
        self.interaction_pairs = []  # List[Dict]: 每个交互对的信息
        self.label_counts = Counter()  # 标签频次统计
        self.co_occurrence_matrix = defaultdict(lambda: defaultdict(int))  # 共现矩阵
        self.scene_label_stats = defaultdict(lambda: defaultdict(int))  # 场景-标签统计
        
        print(f"JRDB Label Analyzer initialized")
        print(f"Data path: {self.data_path}")
        print(f"Social labels path: {self.social_labels_path}")
    
    def load_and_process_data(self):
        """加载并处理所有JRDB社交标注数据"""
        print(f"\n🔄 Loading JRDB social activity labels...")
        
        if not self.social_labels_path.exists():
            raise FileNotFoundError(f"Social labels path not found: {self.social_labels_path}")
        
        json_files = list(self.social_labels_path.glob("*.json"))
        print(f"Found {len(json_files)} scene files")
        
        total_interactions = 0
        
        for json_file in json_files:  # 处理所有文件
            scene_name = json_file.stem
            print(f"Processing scene: {scene_name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)
                
                scene_interactions = self._process_scene_data(scene_data, scene_name)
                total_interactions += scene_interactions
                if scene_interactions > 0:
                    print(f"  Found {scene_interactions} interactions in {scene_name}")
                else:
                    print(f"  No interactions found in {scene_name}")
                
            except Exception as e:
                print(f"  ❌ Error processing {scene_name}: {e}")
                continue
        
        print(f"✅ Data loading completed:")
        print(f"  Total scenes: {len(json_files)}")
        print(f"  Total interactions: {total_interactions}")
        print(f"  Unique interaction types: {len(self.all_interactions)}")
        
        # 调试：显示找到的标签和计数
        print(f"\nTop 10 interaction labels found:")
        for label, count in self.label_counts.most_common(10):
            print(f"  {label}: {count}")
            
        # 调试：显示共现情况
        if self.co_occurrence_matrix:
            print(f"\nSample co-occurrences:")
            sample_count = 0
            for label1, co_labels in self.co_occurrence_matrix.items():
                if co_labels and sample_count < 3:
                    print(f"  {label1} co-occurs with:")
                    for label2, count in list(co_labels.items())[:3]:
                        print(f"    {label2}: {count}")
                    sample_count += 1
        
        return total_interactions
    
    def _process_scene_data(self, scene_data: dict, scene_name: str) -> int:
        """处理单个场景的数据"""
        interactions_count = 0
        
        labels = scene_data.get('labels', {})
        
        for frame_name, frame_data in labels.items():
            # 按人物对聚合交互标签
            person_pair_interactions = defaultdict(list)
            
            for person_data in frame_data:
                h_interactions = person_data.get('H-interaction', [])
                person_id = person_data.get('label_id', '')
                
                for interaction in h_interactions:
                    # 提取交互标签
                    inter_labels = interaction.get('inter_labels', {})
                    active_labels = [label for label, value in inter_labels.items() if value > 0]
                    
                    if not active_labels:
                        continue
                    
                    pair_id = interaction.get('pair', '')
                    
                    # 创建统一的人物对key (保证顺序一致)
                    pair_key = tuple(sorted([person_id, pair_id]))
                    
                    # 将标签添加到这个人物对
                    person_pair_interactions[pair_key].extend(active_labels)
            
            # 处理每个人物对的聚合标签
            for pair_key, all_labels in person_pair_interactions.items():
                # 去重并排序
                unique_labels = list(set(all_labels))
                
                # 记录交互对信息
                interaction_info = {
                    'scene': scene_name,
                    'frame': frame_name,
                    'person_pair': pair_key,
                    'labels': unique_labels,
                    'label_count': len(unique_labels)
                }
                
                self.interaction_pairs.append(interaction_info)
                interactions_count += 1
                
                # 调试：显示多标签情况
                if len(unique_labels) > 1 and interactions_count <= 10:
                    print(f"    Multi-label pair {pair_key}: {unique_labels}")
                
                # 更新统计信息
                self._update_statistics(unique_labels, scene_name)
        
        return interactions_count
    
    def _update_statistics(self, labels: List[str], scene_name: str):
        """更新统计信息"""
        # 标签频次
        for label in labels:
            self.label_counts[label] += 1
            self.all_interactions.add(label)
            self.scene_label_stats[scene_name][label] += 1
        
        # 标签共现矩阵 - 对于多标签情况，每个标签对计数一次
        if len(labels) > 1:  # 只有多标签情况才有共现
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    if i != j:  # 不同标签间的共现
                        self.co_occurrence_matrix[label1][label2] += 1
    
    def calculate_key_metrics(self) -> Dict:
        """计算主要三类的关键指标"""
        print(f"\n📊 Calculating key metrics for main categories...")
        
        metrics = {}
        total_interactions = sum(self.label_counts.values())
        
        for category_key, category_label in self.main_categories.items():
            if category_label not in self.label_counts:
                print(f"  ⚠️ Warning: {category_label} not found in data")
                continue
            
            count = self.label_counts[category_label]
            percentage = (count / total_interactions) * 100
            
            # 计算该标签的共现情况
            co_labels = self.co_occurrence_matrix[category_label]
            total_co_occurrences = sum(co_labels.values())
            
            # 单独出现 vs 与其他标签共现
            solo_appearances = count - total_co_occurrences
            co_occurrence_rate = (total_co_occurrences / count) * 100 if count > 0 else 0
            
            # 最常共现的标签
            top_co_labels = dict(Counter(co_labels).most_common(5))
            
            metrics[category_key] = {
                'label': category_label,
                'count': count,
                'percentage': percentage,
                'solo_appearances': solo_appearances,
                'co_occurrence_rate': co_occurrence_rate,
                'top_co_labels': top_co_labels,
                'total_co_occurrences': total_co_occurrences
            }
            
            print(f"  {category_label}:")
            print(f"    Count: {count:,} ({percentage:.2f}%)")
            print(f"    Co-occurrence rate: {co_occurrence_rate:.2f}%")
            print(f"    Solo appearances: {solo_appearances:,}")
        
        # 三类总覆盖率
        main_three_total = sum(self.label_counts[label] for label in self.main_categories.values() 
                              if label in self.label_counts)
        main_coverage = (main_three_total / total_interactions) * 100
        
        metrics['overall'] = {
            'total_interactions': total_interactions,
            'main_three_coverage': main_coverage,
            'unique_labels': len(self.all_interactions)
        }
        
        print(f"\n📈 Overall Statistics:")
        print(f"  Main three categories coverage: {main_coverage:.2f}%")
        print(f"  Total unique interaction types: {len(self.all_interactions)}")
        
        # 计算标签在interaction中的出现比例
        interaction_label_stats = self._calculate_interaction_label_proportion()
        metrics['interaction_label_stats'] = interaction_label_stats
        
        return metrics
    
    def create_co_occurrence_matrix(self, top_n: int = 15) -> pd.DataFrame:
        """创建标签共现矩阵"""
        print(f"\nCreating co-occurrence matrix (top {top_n} labels)...")
        
        # 选择最频繁的标签
        top_labels = [label for label, _ in self.label_counts.most_common(top_n)]
        print(f"Top labels selected: {top_labels[:5]}...")  # 显示前5个
        
        # 创建共现矩阵
        matrix = np.zeros((len(top_labels), len(top_labels)))
        
        for i, label1 in enumerate(top_labels):
            for j, label2 in enumerate(top_labels):
                if i != j:
                    # 计算条件概率 P(label2|label1)
                    co_count = self.co_occurrence_matrix[label1][label2]
                    total_label1 = self.label_counts[label1]
                    if total_label1 > 0:
                        matrix[i][j] = (co_count / total_label1) * 100
                    
                    # 调试：显示一些共现数据
                    if i < 3 and j < 3 and co_count > 0:
                        print(f"  {label1} -> {label2}: {co_count}/{total_label1} = {matrix[i][j]:.1f}%")
        
        df_matrix = pd.DataFrame(matrix, index=top_labels, columns=top_labels)
        
        # 检查矩阵是否全零
        max_value = df_matrix.values.max()
        print(f"Matrix max value: {max_value:.2f}%")
        if max_value == 0:
            print("Warning: Co-occurrence matrix is all zeros!")
        
        return df_matrix
    
    def plot_heatmap(self, matrix: pd.DataFrame, save_path: str = None):
        """绘制标签共现热力图"""
        print(f"\n🎨 Creating co-occurrence heatmap...")
        
        plt.figure(figsize=(14, 12))
        
        # 创建热力图
        mask = np.diag(np.ones(len(matrix)))  # 隐藏对角线
        
        ax = sns.heatmap(
            matrix, 
            annot=True, 
            fmt='.1f',
            cmap='YlOrRd',
            mask=mask,
            square=True,
            cbar_kws={'label': 'Co-occurrence Rate (%)'},
            annot_kws={'size': 8}
        )
        
        plt.title('JRDB H-Interaction Co-occurrence Matrix\n(Conditional Probability: P(Column|Row))', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Co-occurring Label', fontsize=12, fontweight='bold')
        plt.ylabel('Primary Label', fontsize=12, fontweight='bold')
        
        # 旋转标签以提高可读性
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 突出显示主要三类
        for i, label in enumerate(matrix.index):
            if label in self.main_categories.values():
                ax.add_patch(plt.Rectangle((0, i), len(matrix.columns), 1, 
                                         fill=False, edgecolor='blue', lw=3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Heatmap saved to: {save_path}")
        
        plt.show()
    
    def create_chord_diagram(self, top_n: int = 10, save_path: str = None):
        """创建和弦图"""
        if not CHORD_AVAILABLE:
            print("❌ Chord diagram requires 'chord' package. Skipping...")
            return
        
        print(f"\n🎵 Creating chord diagram (top {top_n} labels)...")
        
        # 选择最频繁的标签
        top_labels = [label for label, _ in self.label_counts.most_common(top_n)]
        
        # 创建共现矩阵数据
        matrix_data = []
        for label1 in top_labels:
            row = []
            for label2 in top_labels:
                if label1 == label2:
                    row.append(0)
                else:
                    # 使用对称的共现计数
                    co_count = (self.co_occurrence_matrix[label1][label2] + 
                               self.co_occurrence_matrix[label2][label1]) / 2
                    row.append(co_count)
            matrix_data.append(row)
        
        # 创建和弦图
        try:
            # 简化标签名称
            short_labels = [label.replace(' together', '').replace('walking', 'walk')
                           .replace('standing', 'stand').replace('sitting', 'sit')
                           .replace('conversation', 'conv') for label in top_labels]
            
            chord = Chord(matrix_data, short_labels)
            chord.to_html(save_path if save_path else "chord_diagram.html")
            print(f"  💾 Chord diagram saved to: {save_path or 'chord_diagram.html'}")
            
        except Exception as e:
            print(f"  ❌ Failed to create chord diagram: {e}")
    
    def analyze_scene_patterns(self) -> Dict:
        """分析不同场景的标签模式"""
        print(f"\n🏢 Analyzing scene-specific patterns...")
        
        scene_analysis = {}
        
        for scene_name, label_stats in self.scene_label_stats.items():
            total_scene_interactions = sum(label_stats.values())
            
            # 计算该场景中主要三类的分布
            main_distribution = {}
            for category_key, category_label in self.main_categories.items():
                count = label_stats.get(category_label, 0)
                percentage = (count / total_scene_interactions) * 100 if total_scene_interactions > 0 else 0
                main_distribution[category_key] = {
                    'count': count,
                    'percentage': percentage
                }
            
            # 找出该场景的特色标签
            scene_total = sum(label_stats.values())
            distinctive_labels = {}
            for label, count in label_stats.items():
                global_rate = (self.label_counts[label] / sum(self.label_counts.values())) * 100
                scene_rate = (count / scene_total) * 100
                if scene_rate > global_rate * 1.5:  # 场景中的比例比全局高50%以上
                    distinctive_labels[label] = {
                        'scene_rate': scene_rate,
                        'global_rate': global_rate,
                        'enrichment': scene_rate / global_rate
                    }
            
            scene_analysis[scene_name] = {
                'total_interactions': total_scene_interactions,
                'main_distribution': main_distribution,
                'distinctive_labels': distinctive_labels
            }
        
        return scene_analysis
    
    def _calculate_interaction_label_proportion(self) -> Dict:
        """计算每个标签在所有interaction中出现的比例"""
        print(f"\n📊 Calculating interaction label proportions...")
        
        # 统计包含每个标签的interaction数量
        label_interaction_counts = defaultdict(int)
        total_interactions = len(self.interaction_pairs)
        
        for interaction in self.interaction_pairs:
            labels_in_interaction = set(interaction['labels'])  # 去重，一个interaction中相同标签只计算一次
            
            for label in labels_in_interaction:
                label_interaction_counts[label] += 1
        
        # 计算比例
        label_proportions = {}
        for label, interaction_count in label_interaction_counts.items():
            proportion = (interaction_count / total_interactions) * 100 if total_interactions > 0 else 0
            label_proportions[label] = {
                'interaction_count': interaction_count,
                'total_interactions': total_interactions,
                'proportion': proportion
            }
        
        # 按比例排序
        sorted_proportions = dict(sorted(label_proportions.items(), 
                                       key=lambda x: x[1]['proportion'], reverse=True))
        
        # 打印Top 10
        print(f"  Top 10 labels by interaction proportion:")
        for i, (label, stats) in enumerate(list(sorted_proportions.items())[:10]):
            print(f"    {i+1:2d}. {label}: {stats['interaction_count']:,}/{total_interactions:,} "
                  f"interactions ({stats['proportion']:.2f}%)")
        
        # 特别关注主要三类
        print(f"\n  Main three categories in interactions:")
        for category_key, category_label in self.main_categories.items():
            if category_label in sorted_proportions:
                stats = sorted_proportions[category_label]
                print(f"    {category_label}: {stats['interaction_count']:,}/{total_interactions:,} "
                      f"interactions ({stats['proportion']:.2f}%)")
            else:
                print(f"    {category_label}: 0 interactions (0.00%)")
        
        return sorted_proportions
    
    def plot_main_categories_analysis(self, metrics: Dict, save_path: str = None):
        """绘制主要三类的分析图表"""
        print(f"\n📊 Creating main categories analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 频率分布饼图
        ax1 = axes[0, 0]
        categories = list(self.main_categories.keys())
        counts = [metrics[cat]['count'] for cat in categories if cat in metrics]
        labels = [metrics[cat]['label'] for cat in categories if cat in metrics]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        if counts:
            wedges, texts, autotexts = ax1.pie(counts, labels=labels, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
        ax1.set_title('Main Categories Distribution\n(by Label Count)', fontsize=12, fontweight='bold')
        
        # 2. 共现率对比
        ax2 = axes[0, 1]
        co_rates = [metrics[cat]['co_occurrence_rate'] for cat in categories if cat in metrics]
        bars = ax2.bar(range(len(co_rates)), co_rates, color=colors[:len(co_rates)])
        ax2.set_xticks(range(len(co_rates)))
        ax2.set_xticklabels([cat.replace('_', ' ').title() for cat in categories if cat in metrics], rotation=45)
        ax2.set_ylabel('Co-occurrence Rate (%)')
        ax2.set_title('Co-occurrence Rates', fontsize=12, fontweight='bold')
        
        # 添加数值标签
        for bar, rate in zip(bars, co_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Interaction比例分析
        ax3 = axes[0, 2]
        if 'interaction_label_stats' in metrics:
            interaction_stats = metrics['interaction_label_stats']
            interaction_props = []
            for cat in categories:
                cat_label = self.main_categories[cat]
                if cat_label in interaction_stats:
                    interaction_props.append(interaction_stats[cat_label]['proportion'])
                else:
                    interaction_props.append(0)
            
            bars3 = ax3.bar(range(len(interaction_props)), interaction_props, color=colors[:len(interaction_props)])
            ax3.set_xticks(range(len(interaction_props)))
            ax3.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45)
            ax3.set_ylabel('Interaction Proportion (%)')
            ax3.set_title('Interaction Coverage\n(% of total interactions)', fontsize=12, fontweight='bold')
            
            # 添加数值标签
            for bar, prop in zip(bars3, interaction_props):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{prop:.1f}%', ha='center', va='bottom')
        
        # 4. 单独 vs 共现出现
        ax4 = axes[1, 0]
        solo_counts = [metrics[cat]['solo_appearances'] for cat in categories if cat in metrics]
        co_counts = [metrics[cat]['total_co_occurrences'] for cat in categories if cat in metrics]
        
        width = 0.35
        x = np.arange(len(solo_counts))
        ax4.bar(x - width/2, solo_counts, width, label='Solo', color='lightblue')
        ax4.bar(x + width/2, co_counts, width, label='Co-occurring', color='orange')
        
        ax4.set_xticks(x)
        ax4.set_xticklabels([cat.replace('_', ' ').title() for cat in categories if cat in metrics], rotation=45)
        ax4.set_ylabel('Count')
        ax4.set_title('Solo vs Co-occurring Appearances', fontsize=12, fontweight='bold')
        ax4.legend()
        
        # 5. 覆盖率分析 (Label vs Interaction)
        ax5 = axes[1, 1]
        if 'overall' in metrics:
            main_coverage = metrics['overall']['main_three_coverage']
            other_coverage = 100 - main_coverage
            
            coverage_data = [main_coverage, other_coverage]
            coverage_labels = ['Main 3 Categories', 'Other Interactions']
            coverage_colors = ['#2ECC71', '#E74C3C']
            
            ax5.pie(coverage_data, labels=coverage_labels, autopct='%1.1f%%', 
                   colors=coverage_colors, startangle=90)
            ax5.set_title('Label Count Coverage', fontsize=12, fontweight='bold')
        
        # 6. 交互覆盖率分析
        ax6 = axes[1, 2]
        if 'interaction_label_stats' in metrics:
            interaction_stats = metrics['interaction_label_stats']
            main_interaction_coverage = 0
            for cat in categories:
                cat_label = self.main_categories[cat]
                if cat_label in interaction_stats:
                    main_interaction_coverage += interaction_stats[cat_label]['proportion']
            
            other_interaction_coverage = 100 - main_interaction_coverage
            
            interaction_coverage_data = [main_interaction_coverage, other_interaction_coverage]
            interaction_coverage_labels = ['Main 3 Categories', 'Other Interactions']
            
            ax6.pie(interaction_coverage_data, labels=interaction_coverage_labels, autopct='%1.1f%%', 
                   colors=coverage_colors, startangle=90)
            ax6.set_title('Interaction Coverage', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Analysis plots saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, metrics: Dict, scene_analysis: Dict, output_path: str = None):
        """生成分析报告"""
        print(f"\n📝 Generating analysis report...")
        
        report = []
        report.append("# JRDB Label Co-occurrence Analysis Report")
        report.append("=" * 50)
        
        # 总体统计
        overall = metrics['overall']
        report.append(f"\n## Overall Statistics")
        report.append(f"- Total interactions: {overall['total_interactions']:,}")
        report.append(f"- Unique interaction types: {overall['unique_labels']}")
        report.append(f"- Main three categories coverage: {overall['main_three_coverage']:.2f}%")
        
        # 主要三类分析
        report.append(f"\n## Main Categories Analysis")
        for category_key, category_data in metrics.items():
            if category_key in ['overall', 'interaction_label_stats']:
                continue
                
            report.append(f"\n### {category_data['label']}")
            report.append(f"- Label count: {category_data['count']:,} ({category_data['percentage']:.2f}%)")
            report.append(f"- Co-occurrence rate: {category_data['co_occurrence_rate']:.2f}%")
            report.append(f"- Solo appearances: {category_data['solo_appearances']:,}")
            
            # 添加interaction比例信息
            if 'interaction_label_stats' in metrics:
                interaction_stats = metrics['interaction_label_stats']
                label_name = category_data['label']
                if label_name in interaction_stats:
                    interaction_info = interaction_stats[label_name]
                    report.append(f"- Interaction coverage: {interaction_info['interaction_count']:,} interactions ({interaction_info['proportion']:.2f}%)")
                else:
                    report.append(f"- Interaction coverage: 0 interactions (0.00%)")
            
            if category_data['top_co_labels']:
                report.append(f"- Top co-occurring labels:")
                for label, count in list(category_data['top_co_labels'].items())[:3]:
                    report.append(f"  - {label}: {count:,}")
        
        # 场景分析摘要
        report.append(f"\n## Scene Analysis Summary")
        report.append(f"- Total scenes analyzed: {len(scene_analysis)}")
        
        # 找出最有特色的场景
        distinctive_scenes = []
        for scene_name, analysis in scene_analysis.items():
            if analysis['distinctive_labels']:
                distinctive_scenes.append((scene_name, len(analysis['distinctive_labels'])))
        
        distinctive_scenes.sort(key=lambda x: x[1], reverse=True)
        
        if distinctive_scenes:
            report.append(f"- Most distinctive scenes:")
            for scene_name, distinctive_count in distinctive_scenes[:5]:
                report.append(f"  - {scene_name}: {distinctive_count} distinctive labels")
        
        # 关键发现
        report.append(f"\n## Key Findings")
        
        # 1. 多级分类合理性
        main_coverage = overall['main_three_coverage']
        if main_coverage > 80:
            report.append(f"✅ **Multi-level classification is well-justified**: Main 3 categories cover {main_coverage:.1f}% of all interactions")
        else:
            report.append(f"⚠️ **Consider expanding main categories**: Only {main_coverage:.1f}% coverage")
        
        # 2. 标签复杂性
        avg_co_occurrence = np.mean([metrics[cat]['co_occurrence_rate'] for cat in self.main_categories.keys() 
                                   if cat in metrics])
        if avg_co_occurrence > 30:
            report.append(f"✅ **Multi-label modeling recommended**: Average co-occurrence rate is {avg_co_occurrence:.1f}%")
        else:
            report.append(f"📝 **Single-label may suffice**: Low co-occurrence rate ({avg_co_occurrence:.1f}%)")
        
        # 3. 场景特异性
        if len(distinctive_scenes) > len(scene_analysis) * 0.3:
            report.append(f"✅ **Scene-specific adaptation needed**: {len(distinctive_scenes)} scenes show distinctive patterns")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"  💾 Report saved to: {output_path}")
        
        print("\n" + report_text)
        return report_text


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='JRDB Label Co-occurrence Analysis')
    
    parser.add_argument('--data_path', type=str, default='../dataset',
                       help='Path to JRDB dataset (contains labels/)')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Output directory for results')
    parser.add_argument('--top_n', type=int, default=15,
                       help='Number of top labels to include in visualizations')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--generate_chord', action='store_true',
                       help='Generate chord diagram (requires chord package)')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"JRDB Label Co-occurrence Analysis")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # 创建分析器
        analyzer = JRDBLabelAnalyzer(args.data_path)
        
        # 加载和处理数据
        total_interactions = analyzer.load_and_process_data()
        
        if total_interactions == 0:
            print("❌ No interaction data found. Please check data path.")
            return
        
        # 计算关键指标
        metrics = analyzer.calculate_key_metrics()
        
        # 创建共现矩阵
        co_matrix = analyzer.create_co_occurrence_matrix(args.top_n)
        
        # 生成可视化
        print(f"\n🎨 Generating visualizations...")
        
        # 热力图
        heatmap_path = output_dir / "co_occurrence_heatmap.png" if args.save_plots else None
        analyzer.plot_heatmap(co_matrix, heatmap_path)
        
        # 主要类别分析图
        analysis_path = output_dir / "main_categories_analysis.png" if args.save_plots else None
        analyzer.plot_main_categories_analysis(metrics, analysis_path)
        
        # 和弦图（可选）
        if args.generate_chord:
            chord_path = output_dir / "chord_diagram.html"
            analyzer.create_chord_diagram(args.top_n, chord_path)
        
        # 场景分析
        scene_analysis = analyzer.analyze_scene_patterns()
        
        # 生成报告
        report_path = output_dir / "analysis_report.md"
        analyzer.generate_report(metrics, scene_analysis, report_path)
        
        # 保存数据
        if args.save_plots:
            # 保存共现矩阵
            matrix_path = output_dir / "co_occurrence_matrix.csv"
            co_matrix.to_csv(matrix_path)
            print(f"💾 Co-occurrence matrix saved to: {matrix_path}")
            
            # 保存指标数据
            import json
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"💾 Metrics saved to: {metrics_path}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"Check results in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()