 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图构建模块
提供构建和操作网络图的功能
"""

import os
import logging
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
from typing import List, Dict, Tuple, Set, Union, Optional

logger = logging.getLogger(__name__)

def build_networkx_graph(entities, relations, include_attributes=True):
    """
    构建NetworkX图
    
    Parameters:
    - entities: 实体列表或DataFrame
    - relations: 关系列表或DataFrame
    - include_attributes: 是否包含所有属性
    
    Returns:
    - NetworkX DiGraph
    """
    logger.info("构建NetworkX图...")
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 转换为DataFrame（如果是列表）
    if isinstance(entities, list):
        entities_df = pd.DataFrame(entities)
    else:
        entities_df = entities
    
    if isinstance(relations, list):
        relations_df = pd.DataFrame(relations)
    else:
        relations_df = relations
    
    # 添加节点
    for _, entity in tqdm(entities_df.iterrows(), desc="添加节点", total=len(entities_df)):
        # 节点ID使用ID字段
        node_id = str(entity['ID'])
        
        # 创建节点属性字典
        attrs = {}
        if include_attributes:
            # 包含所有实体属性
            for col in entities_df.columns:
                if col != 'ID':
                    attrs[col] = entity[col]
        else:
            # 仅包含基本属性
            attrs = {
                'name': entity.get('Name', ''),
                'type': entity.get('Type', ''),
                'is_keyword': entity.get('Is Keyword', False)
            }
        
        # 添加节点
        G.add_node(node_id, **attrs)

    # 添加边
    for _, relation in tqdm(relations_df.iterrows(), desc="添加边", total=len(relations_df)):
        # 获取源节点和目标节点ID
        source_id = str(relation['Source ID'])
        target_id = str(relation['Target ID'])
        
        # 跳过源或目标不在图中的关系
        if source_id not in G.nodes or target_id not in G.nodes:
            continue
        
        # 创建边属性字典
        edge_attrs = {}
        if include_attributes:
            # 包含所有关系属性（除了源和目标ID）
            for col in relations_df.columns:
                if col not in ['Source ID', 'Target ID']:
                    edge_attrs[col] = relation[col]
        else:
            # 仅包含基本属性
            edge_attrs = {
                'relation_type': relation.get('Relation Type', 'Unknown'),
                'confidence': float(relation.get('Confidence', 0.5)),
                'direction': relation.get('Direction', 0)
            }
        
        # 添加边
        G.add_edge(source_id, target_id, **edge_attrs)
    
    logger.info(f"构建了包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边的图")
    return G

def create_visualization_subgraph(G, focal_nodes, max_nodes=1000):
    """创建更健壮的可视化子图，确保包含关键节点和连接"""
    logger.info(f"创建可视化子图，总节点数: {len(G.nodes)}")
    
    # 始终保留关键词实体节点
    nodes_to_keep = set(focal_nodes)
    logger.info(f"关键词节点数: {len(nodes_to_keep)}")
    
    # 如果关键词节点为空，退出
    if not nodes_to_keep:
        logger.warning("未找到关键词节点，尝试包含所有节点")
        # 取前max_nodes个节点作为备选
        all_nodes = list(G.nodes)
        nodes_to_keep = set(all_nodes[:min(max_nodes, len(all_nodes))])
    
    # 添加与关键词节点直接相连的邻居节点
    neighbors = set()
    for node in nodes_to_keep:
        if node in G:  # 确保节点存在于图中
            neighbors.update(G.neighbors(node))
    logger.info(f"发现的邻居节点数: {len(neighbors)}")
    
    # 从邻居中选择要保留的节点
    remaining_slots = max_nodes - len(nodes_to_keep)
    if remaining_slots > 0 and neighbors:
        # 按度排序邻居节点（连接数多的优先）
        neighbor_degrees = [(n, G.degree(n)) for n in neighbors if n not in nodes_to_keep]
        neighbor_degrees.sort(key=lambda x: x[1], reverse=True)
        
        # 添加高度邻居
        top_neighbors = [n for n, _ in neighbor_degrees[:remaining_slots]]
        nodes_to_keep.update(top_neighbors)
    
    # 创建子图
    subgraph = G.subgraph(nodes_to_keep)
    logger.info(f"创建的可视化子图包含 {len(subgraph.nodes)} 个节点和 {len(subgraph.edges)} 条边")
    
    # 如果子图为空，尝试更宽松的方法
    if len(subgraph.nodes) == 0:
        logger.warning("可视化子图为空，尝试更宽松的选择方法")
        # 简单地选取前max_nodes个节点
        all_nodes = list(G.nodes)
        sample_size = min(max_nodes, len(all_nodes))
        subgraph = G.subgraph(all_nodes[:sample_size])
        logger.info(f"备选方法创建的可视化子图包含 {len(subgraph.nodes)} 个节点和 {len(subgraph.edges)} 条边")
    
    return subgraph

def create_drug_synergy_subgraph(G, drug1_id, drug2_id, disease_id, max_intermediate_nodes=50):
    """
    创建药物协同作用子图，重点包含两种药物的共同靶点和通向疾病的路径
    
    Parameters:
    - G: 原始NetworkX图
    - drug1_id: 第一种药物ID
    - drug2_id: 第二种药物ID
    - disease_id: 疾病ID
    - max_intermediate_nodes: 最大中间节点数量
    
    Returns:
    - 协同作用子图
    """
    logger.info(f"创建药物协同子图: 药物1={drug1_id}, 药物2={drug2_id}, 疾病={disease_id}")
    
    # 必须保留的节点
    nodes_to_keep = {drug1_id, drug2_id, disease_id}
    
    # 获取药物1的邻居
    drug1_neighbors = set(G.neighbors(drug1_id)) if drug1_id in G else set()
    
    # 获取药物2的邻居
    drug2_neighbors = set(G.neighbors(drug2_id)) if drug2_id in G else set()
    
    # 获取共同邻居（共同靶点）
    common_neighbors = drug1_neighbors.intersection(drug2_neighbors)
    logger.info(f"发现 {len(common_neighbors)} 个共同靶点")
    
    # 添加共同靶点
    nodes_to_keep.update(common_neighbors)
    
    # 查找从共同靶点到疾病的最短路径
    paths_to_disease = []
    for node in common_neighbors:
        try:
            if nx.has_path(G, node, disease_id):
                path = nx.shortest_path(G, node, disease_id)
                paths_to_disease.append(path)
        except nx.NetworkXError:
            continue
    
    # 添加路径中的所有节点
    for path in paths_to_disease:
        nodes_to_keep.update(path)
    
    # 如果共同靶点少于阈值，添加各自的一些靶点
    if len(common_neighbors) < 5:
        # 按关系强度排序药物1的靶点
        drug1_targets = []
        for target in drug1_neighbors:
            if target in G[drug1_id]:
                confidence = G[drug1_id][target].get('confidence', 0)
                drug1_targets.append((target, confidence))
        
        drug1_targets.sort(key=lambda x: x[1], reverse=True)
        
        # 按关系强度排序药物2的靶点
        drug2_targets = []
        for target in drug2_neighbors:
            if target in G[drug2_id]:
                confidence = G[drug2_id][target].get('confidence', 0)
                drug2_targets.append((target, confidence))
        
        drug2_targets.sort(key=lambda x: x[1], reverse=True)
        
        # 添加一些重要靶点
        top_targets_to_add = min(10, max_intermediate_nodes // 2)
        for target, _ in drug1_targets[:top_targets_to_add]:
            nodes_to_keep.add(target)
        
        for target, _ in drug2_targets[:top_targets_to_add]:
            nodes_to_keep.add(target)
    
    # 限制总节点数
    if len(nodes_to_keep) > max_intermediate_nodes + 3:  # +3 for the drugs and disease
        # 确保药物和疾病节点保留
        essential_nodes = {drug1_id, drug2_id, disease_id}
        other_nodes = list(nodes_to_keep - essential_nodes)
        
        # 随机选择其他节点
        selected_others = random.sample(other_nodes, max_intermediate_nodes)
        nodes_to_keep = set(essential_nodes).union(selected_others)
    
    # 创建子图
    subgraph = G.subgraph(nodes_to_keep)
    logger.info(f"创建的药物协同子图包含 {len(subgraph.nodes)} 个节点和 {len(subgraph.edges)} 条边")
    
    return subgraph

def find_paths_between(G, source_id, target_id, max_length=2):
    """
    查找两个节点之间的所有路径（长度限制）
    
    Parameters:
    - G: NetworkX图
    - source_id: 源节点ID
    - target_id: 目标节点ID
    - max_length: 最大路径长度
    
    Returns:
    - 路径列表
    """
    if source_id not in G or target_id not in G:
        logger.warning(f"源节点 {source_id} 或目标节点 {target_id} 不在图中")
        return []
    
    try:
        # 使用simple_paths查找所有路径
        paths = list(nx.all_simple_paths(G, source_id, target_id, cutoff=max_length))
        logger.info(f"找到 {len(paths)} 条从 {source_id} 到 {target_id} 的路径（最大长度={max_length}）")
        return paths
    except nx.NetworkXError as e:
        logger.error(f"查找路径时出错: {e}")
        return []

def get_node_neighbors_by_type(G, node_id, neighbor_types=None, edge_types=None):
    """
    获取特定类型的邻居节点
    
    Parameters:
    - G: NetworkX图
    - node_id: 节点ID
    - neighbor_types: 邻居类型列表
    - edge_types: 边类型列表
    
    Returns:
    - 满足条件的邻居节点列表
    """
    if node_id not in G:
        logger.warning(f"节点 {node_id} 不在图中")
        return []
    
    neighbors = []
    
    for neighbor in G.neighbors(node_id):
        # 检查邻居类型
        if neighbor_types and G.nodes[neighbor].get('type') not in neighbor_types:
            continue
        
        # 检查边类型
        if edge_types and G[node_id][neighbor].get('relation_type') not in edge_types:
            continue
        
        neighbors.append(neighbor)
    
    return neighbors

def find_common_neighbors(G, node1_id, node2_id, neighbor_types=None):
    """
    查找两个节点的共同邻居
    
    Parameters:
    - G: NetworkX图
    - node1_id: 第一个节点ID
    - node2_id: 第二个节点ID
    - neighbor_types: 邻居类型列表（可选）
    
    Returns:
    - 共同邻居列表
    """
    if node1_id not in G or node2_id not in G:
        logger.warning(f"节点 {node1_id} 或 {node2_id} 不在图中")
        return []
    
    # 获取两个节点的所有邻居
    node1_neighbors = set(G.neighbors(node1_id))
    node2_neighbors = set(G.neighbors(node2_id))
    
    # 计算交集
    common = node1_neighbors.intersection(node2_neighbors)
    
    # 按类型过滤（如果指定）
    if neighbor_types:
        common = {n for n in common if G.nodes[n].get('type') in neighbor_types}
    
    return list(common)