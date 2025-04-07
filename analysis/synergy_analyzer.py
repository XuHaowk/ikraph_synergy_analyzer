#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
药物协同分析模块
分析两种药物之间的协同作用机制
"""

import logging
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Union, Optional

from analysis.psr_engine import PSREngine

logger = logging.getLogger(__name__)

class SynergyAnalyzer:
    """药物协同作用分析器"""
    
    def __init__(self, graph=None, psr_engine=None):
        """
        初始化协同分析器
        
        Parameters:
        - graph: NetworkX图对象（可选）
        - psr_engine: PSR引擎（可选）
        """
        self.graph = graph
        self.psr_engine = psr_engine if psr_engine else PSREngine(graph)
        
        # 结果缓存
        self.results_cache = {}
    
    def set_graph(self, graph):
        """设置图对象"""
        self.graph = graph
        self.psr_engine.set_graph(graph)
        # 清除缓存
        self.results_cache = {}
    
    def find_common_targets(self, drug1_id, drug2_id, target_type="Gene"):
        """
        查找两种药物的共同靶点
        
        Parameters:
        - drug1_id: 第一种药物ID
        - drug2_id: 第二种药物ID
        - target_type: 靶点类型（默认为"Gene"）
        
        Returns:
        - 共同靶点列表
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return []
        
        # 检查药物节点是否存在
        if drug1_id not in self.graph or drug2_id not in self.graph:
            logger.warning(f"药物 {drug1_id} 或 {drug2_id} 不在图中")
            return []
        
        # 获取药物1的靶点
        drug1_targets = []
        for target in self.graph.neighbors(drug1_id):
            if self.graph.nodes[target].get('type') == target_type:
                drug1_targets.append(target)
        
        # 获取药物2的靶点
        drug2_targets = []
        for target in self.graph.neighbors(drug2_id):
            if self.graph.nodes[target].get('type') == target_type:
                drug2_targets.append(target)
        
        # 计算交集
        common_targets = set(drug1_targets).intersection(set(drug2_targets))
        
        logger.info(f"找到 {len(common_targets)} 个共同靶点")
        return list(common_targets)
    
    def analyze_target_regulation(self, drug_id, target_id):
        """
        分析药物对靶点的调节作用
        
        Parameters:
        - drug_id: 药物ID
        - target_id: 靶点ID
        
        Returns:
        - 调节方向: 1(上调/激活), -1(下调/抑制), 0(中性), None(未知)
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return None
        
        # 检查边是否存在
        if not self.graph.has_edge(drug_id, target_id):
            return None
        
        # 获取边属性
        edge_data = self.graph[drug_id][target_id]
        
        # 获取方向信息
        direction = edge_data.get('direction')
        if direction is not None:
            return direction
        
        # 如果没有明确的方向，尝试从关系类型推断
        relation_type = edge_data.get('relation_type', '').lower()
        
        if 'activates' in relation_type or 'increases' in relation_type or 'positive' in relation_type:
            return 1
        elif 'inhibits' in relation_type or 'decreases' in relation_type or 'negative' in relation_type:
            return -1
        else:
            return 0
    
    def calculate_synergy_score(self, drug1_id, drug2_id, disease_id, max_path_length=2):
        """
        计算药物协同评分
        
        Parameters:
        - drug1_id: 第一种药物ID
        - drug2_id: 第二种药物ID
        - disease_id: 疾病ID
        - max_path_length: 最大路径长度
        
        Returns:
        - 协同评分和详细信息
        """
        # 检查缓存
        cache_key = (drug1_id, drug2_id, disease_id, max_path_length)
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        if self.graph is None:
            logger.error("图对象未设置")
            return {"score": 0.0, "mechanisms": [], "targets": []}
        
        # 找到共同靶点
        common_targets = self.find_common_targets(drug1_id, drug2_id)
        
        # 计算每个靶点的协同评分
        target_scores = []
        for target_id in common_targets:
            # 分析药物对靶点的调节作用
            drug1_regulation = self.analyze_target_regulation(drug1_id, target_id)
            drug2_regulation = self.analyze_target_regulation(drug2_id, target_id)
            
            # 如果调节作用未知，跳过
            if drug1_regulation is None or drug2_regulation is None:
                continue
            
            # 分析靶点对疾病的影响
            target_disease_relation = self.psr_engine.infer_indirect_relations(target_id, disease_id, max_path_length)
            target_disease_direction = target_disease_relation["direction"]
            
            if target_disease_direction is None:
                continue
            
            # 计算调节一致性
            regulation_consistency = 1 if (drug1_regulation == drug2_regulation) else -1
            
            # 计算与疾病相关性
            disease_relevance = target_disease_relation["probability"]
            
            # 组合评分
            target_score = regulation_consistency * drug1_regulation * drug2_regulation * target_disease_direction * disease_relevance
            
            target_scores.append({
                "target_id": target_id,
                "target_name": self.graph.nodes[target_id].get("name", ""),
                "drug1_regulation": drug1_regulation,
                "drug2_regulation": drug2_regulation,
                "disease_direction": target_disease_direction,
                "disease_relevance": disease_relevance,
                "score": target_score
            })
        
        # 找到药物特有的靶点
        drug1_targets = set(self.graph.neighbors(drug1_id)) - set(common_targets)
        drug2_targets = set(self.graph.neighbors(drug2_id)) - set(common_targets)
        
        # 分析药物特有靶点的互补性
        complementary_mechanisms = []
        
        # 分析药物1特有靶点
        for target1_id in drug1_targets:
            if self.graph.nodes[target1_id].get('type') != "Gene":
                continue
                
            # 分析药物对靶点的调节作用
            drug1_regulation = self.analyze_target_regulation(drug1_id, target1_id)
            
            if drug1_regulation is None:
                continue
            
            # 分析靶点对疾病的影响
            target1_disease_relation = self.psr_engine.infer_indirect_relations(target1_id, disease_id, max_path_length)
            target1_disease_direction = target1_disease_relation["direction"]
            
            if target1_disease_direction is None:
                continue
            
            # 寻找药物2的互补靶点
            for target2_id in drug2_targets:
                if self.graph.nodes[target2_id].get('type') != "Gene":
                    continue
                
                # 分析药物对靶点的调节作用
                drug2_regulation = self.analyze_target_regulation(drug2_id, target2_id)
                
                if drug2_regulation is None:
                    continue
                
                # 分析靶点对疾病的影响
                target2_disease_relation = self.psr_engine.infer_indirect_relations(target2_id, disease_id, max_path_length)
                target2_disease_direction = target2_disease_relation["direction"]
                
                if target2_disease_direction is None:
                    continue
                
                # 分析两个靶点之间的关系
                target_interaction = self.psr_engine.infer_indirect_relations(target1_id, target2_id, 1)
                
                # 如果两个靶点有直接关系，计算互补性
                if target_interaction["probability"] > 0:
                    # 计算互补性评分
                    complementary_score = (
                        drug1_regulation * target1_disease_direction * 
                        drug2_regulation * target2_disease_direction * 
                        target_interaction["probability"]
                    )
                    
                    complementary_mechanisms.append({
                        "target1_id": target1_id,
                        "target1_name": self.graph.nodes[target1_id].get("name", ""),
                        "target2_id": target2_id,
                        "target2_name": self.graph.nodes[target2_id].get("name", ""),
                        "drug1_regulation": drug1_regulation,
                        "drug2_regulation": drug2_regulation,
                        "interaction_probability": target_interaction["probability"],
                        "score": complementary_score
                    })
        
        # 计算总体协同评分
        total_score = 0.0
        
        # 共同靶点贡献
        if target_scores:
            common_target_score = sum(item["score"] for item in target_scores) / len(target_scores)
            total_score += common_target_score * 0.6  # 60%权重
        
        # 互补机制贡献
        if complementary_mechanisms:
            complementary_score = sum(item["score"] for item in complementary_mechanisms) / len(complementary_mechanisms)
            total_score += complementary_score * 0.4  # 40%权重
        
        # 准备结果
        result = {
            "score": total_score,
            "common_targets": target_scores,
            "complementary_mechanisms": complementary_mechanisms,
            "summary": {
                "common_target_count": len(target_scores),
                "complementary_mechanism_count": len(complementary_mechanisms)
            }
        }
        
        # 缓存结果
        self.results_cache[cache_key] = result
        
        return result
    
    def rank_synergy_candidates(self, drug_id, disease_id, candidate_drugs, max_path_length=2):
        """
        对协同药物候选进行排序
        
        Parameters:
        - drug_id: 基准药物ID
        - disease_id: 疾病ID
        - candidate_drugs: 候选药物ID列表
        - max_path_length: 最大路径长度
        
        Returns:
        - 按协同评分排序的候选药物列表
        """
        rankings = []
        
        for candidate_id in tqdm(candidate_drugs, desc="评估候选药物"):
            # 计算协同评分
            synergy_result = self.calculate_synergy_score(drug_id, candidate_id, disease_id, max_path_length)
            
            # 添加到排名列表
            rankings.append({
                "drug_id": candidate_id,
                "drug_name": self.graph.nodes[candidate_id].get("name", "") if self.graph else "",
                "synergy_score": synergy_result["score"],
                "common_target_count": synergy_result["summary"]["common_target_count"],
                "complementary_mechanism_count": synergy_result["summary"]["complementary_mechanism_count"]
            })
        
        # 按评分排序
        rankings.sort(key=lambda x: x["synergy_score"], reverse=True)
        
        return rankings 
