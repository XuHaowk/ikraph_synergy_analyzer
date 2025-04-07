#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图操作工具模块
提供图操作相关的工具函数
"""

import logging
import networkx as nx
import pandas as pd
import random
from typing import List, Dict, Tuple, Set, Union, Optional

logger = logging.getLogger(__name__)

def merge_graphs(graphs):
    """
    合并多个图
    
    Parameters:
    - graphs: 图对象列表
    
    Returns:
    - 合并后的图
    """
    if not graphs:
        logger.warning("没有图需要合并")
        return nx.DiGraph()
    
    # 创建新图
    merged_graph = nx.DiGraph()
    
    # 合并节点和边
    for G in graphs:
        # 添加节点
        for node, attrs in G.nodes(data=True):
            if node not in merged_graph:
                merged_graph.add_node(node, **attrs)
        
        # 添加边
        for source, target, attrs in G.edges(data=True):
            if not merged_graph.has_edge(source, target):
                merged_graph.add_edge(source, target, **attrs)
    
    logger.info(f"合并了 {len(graphs)} 个图，结果图包含 {len(merged_graph.nodes)} 个节点和 {len(merged_graph.edges)} 条边")
    return merged_graph

def find_paths(G, source_id, target_id, max_length=2):
    """
    查找两个节点之间的所有路径
    
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

def calculate_centrality(G, method='degree'):
    """
    计算图中节点的中心性
    
    Parameters:
    - G: NetworkX图
    - method: 中心性计算方法，可选值：'degree', 'betweenness', 'closeness', 'eigenvector'
    
    Returns:
    - 节点中心性字典
    """
    if len(G.nodes) == 0:
        logger.warning("图中没有节点")
        return {}
    
    if method == 'degree':
        centrality = nx.degree_centrality(G)
    elif method == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif method == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif method == 'eigenvector':
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
    else:
        logger.error(f"不支持的中心性计算方法: {method}")
        return {}
    
    return centrality

def get_subgraph_by_node_types(G, node_types):
    """
    根据节点类型获取子图
    
    Parameters:
    - G: NetworkX图
    - node_types: 节点类型列表
    
    Returns:
    - 子图
    """
    if not node_types:
        logger.warning("未指定节点类型")
        return G
    
    # 筛选节点
    nodes = [node for node, attrs in G.nodes(data=True) 
            if attrs.get('type') in node_types]
    
    # 创建子图
    subgraph = G.subgraph(nodes)
    logger.info(f"创建了包含 {len(subgraph.nodes)} 个节点和 {len(subgraph.edges)} 条边的子图（节点类型: {node_types}）")
    
    return subgraph

def get_subgraph_by_relation_types(G, relation_types):
    """
    根据关系类型获取子图
    
    Parameters:
    - G: NetworkX图
    - relation_types: 关系类型列表
    
    Returns:
    - 子图
    """
    if not relation_types:
        logger.warning("未指定关系类型")
        return G
    
    # 创建新图
    subgraph = nx.DiGraph()
    
    # 添加所有节点
    for node, attrs in G.nodes(data=True):
        subgraph.add_node(node, **attrs)
    
    # 筛选边
    for source, target, attrs in G.edges(data=True):
        relation_type = attrs.get('relation_type')
        if relation_type in relation_types:
            subgraph.add_edge(source, target, **attrs)
    
    # 移除没有边的节点
    isolated_nodes = list(nx.isolates(subgraph))
    subgraph.remove_nodes_from(isolated_nodes)
    
    logger.info(f"创建了包含 {len(subgraph.nodes)} 个节点和 {len(subgraph.edges)} 条边的子图（关系类型: {relation_types}）")
    
    return subgraph

def sample_graph(G, max_nodes=1000, prioritize_nodes=None):
    """
    对图进行采样
    
    Parameters:
    - G: NetworkX图
    - max_nodes: 最大节点数
    - prioritize_nodes: 优先保留的节点列表
    
    Returns:
    - 采样后的图
    """
    if len(G.nodes) <= max_nodes:
        return G
    
    # 确保优先节点存在
    if prioritize_nodes:
        priority_nodes = [n for n in prioritize_nodes if n in G]
    else:
        priority_nodes = []
    
    # 计算剩余节点数
    remaining = max_nodes - len(priority_nodes)
    
    if remaining <= 0:
        # 如果优先节点超过最大节点数，只保留一部分
        if len(priority_nodes) > max_nodes:
            nodes_to_keep = priority_nodes[:max_nodes]
        else:
            nodes_to_keep = priority_nodes
    else:
        # 随机选择其他节点
        other_nodes = list(set(G.nodes) - set(priority_nodes))
        selected_others = random.sample(other_nodes, min(remaining, len(other_nodes)))
        nodes_to_keep = priority_nodes + selected_others
    
    # 创建子图
    sampled_graph = G.subgraph(nodes_to_keep)
    logger.info(f"采样图包含 {len(sampled_graph.nodes)} 个节点和 {len(sampled_graph.edges)} 条边")
    
    return sampled_graph 
