#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化工具模块
提供图形可视化相关的工具函数
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Set, Union, Optional

logger = logging.getLogger(__name__)

# 尝试导入可选的可视化库
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    logger.warning("未找到pyvis库，某些可视化功能可能不可用")
    PYVIS_AVAILABLE = False

def generate_network_visualization(G, output_file, title="网络图", height="800px", width="100%"):
    """
    生成网络可视化
    
    Parameters:
    - G: NetworkX图
    - output_file: 输出文件路径
    - title: 标题
    - height: 高度
    - width: 宽度
    
    Returns:
    - 输出文件路径
    """
    if not PYVIS_AVAILABLE:
        logger.error("生成交互式网络图需要pyvis库。请使用 'pip install pyvis' 安装")
        return None
    
    if len(G.nodes) == 0:
        logger.warning("图中没有节点，无法生成可视化")
        return None
    
    try:
        # 创建Network对象
        net = Network(height=height, width=width, directed=True, notebook=False)
        net.heading = title
        
        # 配置节点颜色函数
        def get_node_color(node_type, is_keyword=False):
            if node_type == "Chemical" or node_type == "Drug":
                return "#1f77b4" if is_keyword else "#aec7e8"  # 蓝色/浅蓝色
            elif node_type == "Disease":
                return "#d62728" if is_keyword else "#ff9896"  # 红色/浅红色
            elif node_type == "Gene" or node_type == "Protein":
                return "#2ca02c"  # 绿色
            elif node_type == "Pathway":
                return "#9467bd"  # 紫色
            else:
                return "#7f7f7f"  # 灰色
        
        # 添加节点
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get("type", "Unknown")
            is_keyword = attrs.get("is_keyword", False)
            
            # 节点颜色
            color = get_node_color(node_type, is_keyword)
            
            # 节点大小
            size = 30 if is_keyword else 20 if node_type in ["Chemical", "Disease", "Drug"] else 15
            
            # 节点标题（鼠标悬停显示）
            title = f"<b>{node_type}: {attrs.get('name', '')}</b><br>"
            if "External ID" in attrs and attrs["External ID"]:
                title += f"ID: {attrs['External ID']}<br>"
            
            # 添加节点
            net.add_node(
                node, 
                label=attrs.get("name", ""),
                title=title,
                color=color,
                size=size
            )
        
        # 添加边
        for source, target, attrs in G.edges(data=True):
            # 边标题
            edge_title = f"<b>{attrs.get('relation_type', 'Unknown')}</b><br>"
            if "confidence" in attrs:
                edge_title += f"Confidence: {attrs.get('confidence', 0):.2f}<br>"
            if "direction" in attrs:
                direction = attrs.get("direction")
                if direction == 1:
                    edge_title += "Direction: Positive<br>"
                elif direction == -1:
                    edge_title += "Direction: Negative<br>"
            
            # 边宽度和颜色
            width = 1 + 2 * attrs.get("confidence", 0.5)
            
            # 如果有方向信息，边颜色可以显示方向
            if "direction" in attrs:
                direction = attrs.get("direction")
                if direction == 1:
                    color = "#1f77b4"  # 蓝色：正向
                elif direction == -1:
                    color = "#d62728"  # 红色：负向
                else:
                    color = "#7f7f7f"  # 灰色：中性
            else:
                color = "#7f7f7f"
            
            # 添加边
            net.add_edge(
                source, 
                target,
                title=edge_title,
                width=width,
                color=color
            )
        
        # 配置物理引擎
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000
                }
            },
            "interaction": {
                "navigationButtons": true,
                "keyboard": true,
                "tooltipDelay": 300
            }
        }
        """)
        
        # 保存网络
        net.save_graph(output_file)
        logger.info(f"保存网络可视化到 {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"生成网络可视化时出错: {e}")
        return None

def generate_synergy_heatmap(synergy_scores, output_file, title="药物协同热图"):
    """
    生成药物协同热图
    
    Parameters:
    - synergy_scores: 协同评分字典，格式为{(drug1, drug2): score}
    - output_file: 输出文件路径
    - title: 标题
    
    Returns:
    - 输出文件路径
    """
    if not synergy_scores:
        logger.warning("没有协同评分数据，无法生成热图")
        return None
    
    try:
        # 获取所有药物
        drugs = set()
        for drug1, drug2 in synergy_scores:
            drugs.add(drug1)
            drugs.add(drug2)
        
        drugs = sorted(list(drugs))
        n_drugs = len(drugs)
        
        # 创建热图矩阵
        heatmap_data = np.zeros((n_drugs, n_drugs))
        
        # 填充矩阵
        for i, drug1 in enumerate(drugs):
            for j, drug2 in enumerate(drugs):
                if i == j:
                    heatmap_data[i, j] = 0
                else:
                    # 尝试两种顺序，返回存在的那个
                    key1 = (drug1, drug2)
                    key2 = (drug2, drug1)
                    
                    if key1 in synergy_scores:
                        heatmap_data[i, j] = synergy_scores[key1]
                    elif key2 in synergy_scores:
                        heatmap_data[i, j] = synergy_scores[key2]
                    else:
                        heatmap_data[i, j] = 0
        
        # 创建热图
        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap_data, cmap='viridis')
        
        # 添加标签
        plt.colorbar(label='协同评分')
        plt.title(title)
        plt.xticks(range(n_drugs), drugs, rotation=90)
        plt.yticks(range(n_drugs), drugs)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        logger.info(f"保存药物协同热图到 {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"生成药物协同热图时出错: {e}")
        return None

def generate_pathway_network(pathways, output_file, title="通路网络"):
    """
    生成通路网络图
    
    Parameters:
    - pathways: 通路列表，每个通路是一个包含'pathway_id', 'pathway_name', 'genes'等字段的字典
    - output_file: 输出文件路径
    - title: 标题
    
    Returns:
    - 输出文件路径
    """
    if not pathways:
        logger.warning("没有通路数据，无法生成网络图")
        return None

    try:
        # 创建通路网络图
        G = nx.Graph()
        
        # 添加通路节点
        for pathway in pathways:
            G.add_node(
                pathway['pathway_id'],
                name=pathway['pathway_name'],
                type='Pathway',
                gene_count=len(pathway.get('genes', []))
            )
        
        # 添加通路之间的连接
        for i, p1 in enumerate(pathways):
            for j, p2 in enumerate(pathways[i+1:], i+1):
                # 计算基因重叠
                genes1 = set(p1.get('genes', []))
                genes2 = set(p2.get('genes', []))
                
                if not genes1 or not genes2:
                    continue
                
                # 计算Jaccard相似度
                overlap = len(genes1.intersection(genes2))
                union = len(genes1.union(genes2))
                
                if overlap > 0:
                    similarity = overlap / union
                    # 添加边
                    G.add_edge(
                        p1['pathway_id'],
                        p2['pathway_id'],
                        weight=similarity,
                        overlap=overlap
                    )
        
        # 可视化
        if PYVIS_AVAILABLE:
            # 使用pyvis创建交互式网络图
            net = Network(height="800px", width="100%", notebook=False)
            
            # 配置节点大小比例尺
            min_genes = min(pathway.get('gene_count', 1) for pathway in pathways)
            max_genes = max(pathway.get('gene_count', 1) for pathway in pathways)
            
            # 添加节点
            for pathway in pathways:
                # 计算节点大小
                gene_count = pathway.get('gene_count', 1)
                size = 10 + (gene_count - min_genes) * 20 / max(1, max_genes - min_genes)
                
                net.add_node(
                    pathway['pathway_id'],
                    label=pathway['pathway_name'],
                    title=f"<b>{pathway['pathway_name']}</b><br>Genes: {gene_count}",
                    color="#9467bd",  # 紫色表示通路
                    size=int(size)
                )
            
            # 添加边
            for u, v, attrs in G.edges(data=True):
                net.add_edge(
                    u, v,
                    title=f"Shared genes: {attrs.get('overlap')}<br>Similarity: {attrs.get('weight'):.3f}",
                    width=1 + 9 * attrs.get('weight', 0),
                    color={"color": "rgba(200,200,200,0.5)", "opacity": 0.8}
                )
            
            # 设置物理引擎
            net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -100,
                        "centralGravity": 0.05,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "solver": "forceAtlas2Based",
                    "stabilization": {
                        "enabled": true,
                        "iterations": 1000
                    }
                }
            }
            """)
            
            # 保存交互式网络图
            net.save_graph(output_file)
        else:
            # 使用matplotlib创建静态网络图
            plt.figure(figsize=(12, 10))
            
            # 计算节点大小
            node_sizes = []
            for node in G.nodes():
                gene_count = G.nodes[node].get('gene_count', 1)
                size = 100 + gene_count * 10
                node_sizes.append(size)
            
            # 计算边宽度
            edge_widths = []
            for u, v in G.edges():
                weight = G[u][v].get('weight', 0)
                width = 1 + 5 * weight
                edge_widths.append(width)
            
            # 使用spring_layout进行布局
            pos = nx.spring_layout(G, seed=42)
            
            # 绘制网络
            nx.draw_networkx_nodes(G, pos, node_color='purple', node_size=node_sizes, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
            nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['name'] for node in G.nodes()}, font_size=8)
            
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
        logger.info(f"保存通路网络图到 {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"生成通路网络图时出错: {e}")
        return None

def generate_mechanism_diagram(mechanisms, output_file, title="作用机制图"):
    """
    生成作用机制示意图
    
    Parameters:
    - mechanisms: 机制列表，每个机制是一个字典
    - output_file: 输出文件路径
    - title: 标题
    
    Returns:
    - 输出文件路径
    """
    if not mechanisms:
        logger.warning("没有机制数据，无法生成示意图")
        return None
    
    try:
        # 创建图形
        plt.figure(figsize=(14, 10))
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 收集所有节点
        all_nodes = set()
        
        for mechanism in mechanisms:
            # 提取药物、基因和毒性/疾病节点
            drug_id = mechanism.get('drug_id', 'Unknown')
            drug_name = mechanism.get('drug_name', 'Unknown Drug')
            
            gene_id = mechanism.get('gene_id', 'Unknown')
            gene_name = mechanism.get('gene_name', 'Unknown Gene')
            
            target_id = mechanism.get('target_id', mechanism.get('toxicity_id', 'Unknown'))
            target_name = mechanism.get('target_name', mechanism.get('toxicity_name', 'Unknown Target'))
            
            # 添加节点
            G.add_node(drug_id, name=drug_name, type='Drug')
            G.add_node(gene_id, name=gene_name, type='Gene')
            G.add_node(target_id, name=target_name, type='Target')
            
            # 提取关系方向
            drug_gene_direction = mechanism.get('drug_gene_direction', 0)
            gene_target_direction = mechanism.get('gene_target_direction', 
                                              mechanism.get('gene_toxicity_direction', 0))
            
            # 添加边
            G.add_edge(drug_id, gene_id, direction=drug_gene_direction)
            G.add_edge(gene_id, target_id, direction=gene_target_direction)
            
            # 收集节点
            all_nodes.add((drug_id, drug_name, 'Drug'))
            all_nodes.add((gene_id, gene_name, 'Gene'))
            all_nodes.add((target_id, target_name, 'Target'))
        
        # 使用分层布局
        pos = nx.multipartite_layout(G, subset_key='type')
        
        # 绘制节点
        drug_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'Drug']
        gene_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'Gene']
        target_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'Target']
        
        plt.subplot(1, 1, 1)
        nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='blue', node_size=500, alpha=0.8, label='Drugs')
        nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color='green', node_size=500, alpha=0.8, label='Genes')
        nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='red', node_size=500, alpha=0.8, label='Targets')
        
        # 绘制边，不同颜色表示不同方向
        positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction', 0) > 0]
        negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction', 0) < 0]
        neutral_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('direction', 0) == 0]
        
        nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color='green', arrows=True, width=2, label='Activation')
        nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color='red', arrows=True, width=2, label='Inhibition')
        nx.draw_networkx_edges(G, pos, edgelist=neutral_edges, edge_color='gray', arrows=True, width=1, label='Unknown')
        
        # 添加节点标签
        labels = {node: G.nodes[node]['name'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        plt.title(title)
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存作用机制图到 {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"生成作用机制图时出错: {e}")
        return None
