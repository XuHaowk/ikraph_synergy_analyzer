 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
概率语义推理（PSR）引擎
实现基于概率的语义推理算法，用于间接关系推断
"""

import logging
import numpy as np
from tqdm import tqdm
import networkx as nx
from typing import List, Dict, Tuple, Set, Union, Optional

logger = logging.getLogger(__name__)

class PSREngine:
    """概率语义推理引擎"""
    
    def __init__(self, graph=None, relation_schema=None):
        """
        初始化PSR引擎
        
        Parameters:
        - graph: NetworkX图对象（可选）
        - relation_schema: 关系模式（可选）
        """
        self.graph = graph
        self.relation_schema = relation_schema
        
        # 直接关系概率缓存
        self.direct_prob_cache = {}
        
        # 间接关系概率缓存
        self.indirect_prob_cache = {}
    
    def set_graph(self, graph):
        """设置图对象"""
        self.graph = graph
        # 清除缓存
        self.direct_prob_cache = {}
        self.indirect_prob_cache = {}
    
    def set_relation_schema(self, relation_schema):
        """设置关系模式"""
        self.relation_schema = relation_schema
    
    def calculate_direct_probability(self, source_id, target_id, relation_type=None):
        """
        计算两个实体之间的直接关系概率
        
        基于公式:
        P_{A→B} = 1 - ∏_{i=1}^{N}(1 - p^i_{A→B})
        
        Parameters:
        - source_id: 源实体ID
        - target_id: 目标实体ID
        - relation_type: 关系类型（可选）
        
        Returns:
        - 直接关系的概率
        """
        # 检查缓存
        cache_key = (source_id, target_id, relation_type)
        if cache_key in self.direct_prob_cache:
            return self.direct_prob_cache[cache_key]
        
        if self.graph is None:
            logger.error("图对象未设置")
            return 0.0
        
        # 检查节点是否存在
        if source_id not in self.graph or target_id not in self.graph:
            return 0.0
        
        # 检查边是否存在
        if not self.graph.has_edge(source_id, target_id):
            return 0.0
        
        # 获取边属性
        edge_data = self.graph[source_id][target_id]
        
        # 如果指定了关系类型，检查是否匹配
        if relation_type and edge_data.get('relation_type') != relation_type:
            return 0.0
        
        # 获取置信度
        confidence = edge_data.get('confidence', 0.0)
        if not confidence:
            # 尝试获取其他概率相关字段
            confidence = edge_data.get('probability', edge_data.get('weight', 0.5))
        
        # 储存到缓存
        self.direct_prob_cache[cache_key] = float(confidence)
        
        return float(confidence)
    
    def calculate_indirect_probability(self, source_id, target_id, via_node_id, relation_types=None):
        """
        计算通过中间节点的间接关系概率
        
        基于公式:
        P_{A→C via B} = P_{A→B} × P_{B→C}
        
        Parameters:
        - source_id: 源实体ID
        - target_id: 目标实体ID
        - via_node_id: 中间节点ID
        - relation_types: 关系类型列表（可选）
        
        Returns:
        - 间接关系的概率
        """
        # 计算 A→B 的概率
        p_a_b = self.calculate_direct_probability(source_id, via_node_id, 
                                                relation_types[0] if relation_types and len(relation_types) > 0 else None)
        
        # 计算 B→C 的概率
        p_b_c = self.calculate_direct_probability(via_node_id, target_id,
                                                relation_types[1] if relation_types and len(relation_types) > 1 else None)
        
        # 计算间接关系概率
        return p_a_b * p_b_c
    
    def calculate_indirect_probability_all_paths(self, source_id, target_id, intermediate_nodes, relation_types=None):
        """
        计算所有中间路径的间接关系概率
        
        基于公式:
        P_{A→C} = 1 - ∏_{i=1}^{m}(1 - P_{A→B_i→C})
        
        Parameters:
        - source_id: 源实体ID
        - target_id: 目标实体ID
        - intermediate_nodes: 中间节点ID列表
        - relation_types: 关系类型列表（可选）
        
        Returns:
        - 通过所有中间节点的间接关系概率
        """
        # 检查缓存
        cache_key = (source_id, target_id, tuple(intermediate_nodes), tuple(relation_types) if relation_types else None)
        if cache_key in self.indirect_prob_cache:
            return self.indirect_prob_cache[cache_key]
        
        if not intermediate_nodes:
            return 0.0
        
        # 计算通过每个中间节点的概率
        path_probs = []
        for node_id in intermediate_nodes:
            prob = self.calculate_indirect_probability(source_id, target_id, node_id, relation_types)
            if prob > 0:
                path_probs.append(prob)
        
        if not path_probs:
            return 0.0
        
        # 计算整体概率
        overall_prob = 1.0 - np.prod([1.0 - p for p in path_probs])
        
        # 储存到缓存
        self.indirect_prob_cache[cache_key] = overall_prob
        
        return overall_prob
    
    def determine_relation_direction(self, source_id, target_id, intermediate_nodes):
        """
        确定间接关系的方向
        
        Parameters:
        - source_id: 源实体ID
        - target_id: 目标实体ID
        - intermediate_nodes: 中间节点ID列表
        
        Returns:
        - 关系方向: 1(正相关), -1(负相关), 0(中性), None(未知)
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return None
        
        directions = []
        
        for node_id in intermediate_nodes:
            if self.graph.has_edge(source_id, node_id) and self.graph.has_edge(node_id, target_id):
                # 获取方向信息
                dir_a_b = self.graph[source_id][node_id].get('direction', 0)
                dir_b_c = self.graph[node_id][target_id].get('direction', 0)
                
                # 计算复合方向
                if dir_a_b != 0 and dir_b_c != 0:
                    # 正负相乘
                    compound_dir = dir_a_b * dir_b_c
                    directions.append(compound_dir)
        
        if not directions:
            return None
        
        # 取多数方向
        pos_count = directions.count(1)
        neg_count = directions.count(-1)
        
        if pos_count > neg_count:
            return 1
        elif neg_count > pos_count:
            return -1
        else:
            return 0
    
    def find_all_paths(self, source_id, target_id, max_length=2):
        """
        查找两个节点之间的所有路径
        
        Parameters:
        - source_id: 源节点ID
        - target_id: 目标节点ID
        - max_length: 最大路径长度
        
        Returns:
        - 路径列表
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return []
        
        if source_id not in self.graph or target_id not in self.graph:
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_length))
            return paths
        except nx.NetworkXError:
            return []
    
    def infer_indirect_relations(self, source_id, target_id, max_path_length=2):
        """
        推断间接关系
        
        Parameters:
        - source_id: 源实体ID
        - target_id: 目标实体ID
        - max_path_length: 最大路径长度
        
        Returns:
        - 字典，包含概率、方向、中间路径等信息
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return {"probability": 0.0, "direction": None, "paths": []}
        
        # 查找所有路径
        paths = self.find_all_paths(source_id, target_id, max_path_length)
        
        if not paths:
            return {"probability": 0.0, "direction": None, "paths": []}
        
        # 提取所有中间节点
        intermediate_nodes = set()
        for path in paths:
            if len(path) > 2:  # 至少有一个中间节点
                intermediate_nodes.update(path[1:-1])
        
        # 计算概率
        probability = self.calculate_indirect_probability_all_paths(
            source_id, target_id, list(intermediate_nodes))
        
        # 确定方向
        direction = self.determine_relation_direction(
            source_id, target_id, intermediate_nodes)
        
        return {
            "probability": probability,
            "direction": direction,
            "paths": paths,
            "intermediate_nodes": list(intermediate_nodes)
        }
    
    def batch_infer_relations(self, entity_pairs, max_path_length=2):
        """
        批量推断关系
        
        Parameters:
        - entity_pairs: 实体对列表，每对为(source_id, target_id)
        - max_path_length: 最大路径长度
        
        Returns:
        - 字典，键为实体对，值为推断结果
        """
        results = {}
        
        for source_id, target_id in tqdm(entity_pairs, desc="推断关系"):
            result = self.infer_indirect_relations(source_id, target_id, max_path_length)
            results[(source_id, target_id)] = result
        
        return results