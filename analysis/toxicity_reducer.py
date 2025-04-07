#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
毒性减轻分析模块
分析一种药物如何减轻另一种药物的毒性
"""

import logging
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Union, Optional

from analysis.psr_engine import PSREngine

logger = logging.getLogger(__name__)

class ToxicityReducer:
    """毒性减轻分析器"""
    
    def __init__(self, graph=None, psr_engine=None):
        """
        初始化毒性减轻分析器
        
        Parameters:
        - graph: NetworkX图对象（可选）
        - psr_engine: PSR引擎（可选）
        """
        self.graph = graph
        self.psr_engine = psr_engine if psr_engine else PSREngine(graph)
        
        # 缓存
        self.drug_toxicity_cache = {}
        self.protection_cache = {}
    
    def set_graph(self, graph):
        """设置图对象"""
        self.graph = graph
        self.psr_engine.set_graph(graph)
        # 清除缓存
        self.drug_toxicity_cache = {}
        self.protection_cache = {}
    
    def identify_toxicity_genes(self, drug_id, toxicity_terms=None):
        """
        识别药物毒性相关基因
        
        Parameters:
        - drug_id: 药物ID
        - toxicity_terms: 毒性相关术语列表（可选）
        
        Returns:
        - 毒性相关基因列表
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return []
        
        # 检查缓存
        cache_key = (drug_id, tuple(toxicity_terms) if toxicity_terms else None)
        if cache_key in self.drug_toxicity_cache:
            return self.drug_toxicity_cache[cache_key]
        
        # 默认毒性术语
        if toxicity_terms is None:
            toxicity_terms = ["toxicity", "side effect", "adverse", "damage", "injury"]
        
        # 收集毒性相关的节点
        toxicity_nodes = []
        for node, attrs in self.graph.nodes(data=True):
            node_name = attrs.get('name', '').lower()
            if any(term in node_name for term in toxicity_terms):
                toxicity_nodes.append(node)
        
        # 检查药物与毒性节点的关系
        toxicity_genes = []
        for toxicity_node in toxicity_nodes:
            # 尝试找到药物到毒性节点的路径
            paths = self.psr_engine.find_all_paths(drug_id, toxicity_node, 2)
            
            for path in paths:
                if len(path) == 3:  # 药物-基因-毒性
                    gene_id = path[1]
                    if self.graph.nodes[gene_id].get('type') == "Gene":
                        # 分析药物对基因的调节
                        if self.graph.has_edge(drug_id, gene_id):
                            drug_gene_direction = self.graph[drug_id][gene_id].get('direction', 0)
                        else:
                            drug_gene_direction = 0
                        
                        # 分析基因对毒性的影响
                        if self.graph.has_edge(gene_id, toxicity_node):
                            gene_toxicity_direction = self.graph[gene_id][toxicity_node].get('direction', 0)
                        else:
                            gene_toxicity_direction = 0
                        
                        # 只有当调节方向一致（都是正向或都是负向）时，基因才被认为与毒性相关
                        if drug_gene_direction * gene_toxicity_direction > 0:
                            toxicity_genes.append({
                                "gene_id": gene_id,
                                "gene_name": self.graph.nodes[gene_id].get('name', ''),
                                "toxicity_node": toxicity_node,
                                "toxicity_name": self.graph.nodes[toxicity_node].get('name', ''),
                                "drug_gene_direction": drug_gene_direction,
                                "gene_toxicity_direction": gene_toxicity_direction,
                                "toxicity_contribution": drug_gene_direction * gene_toxicity_direction
                            })
        
        # 缓存结果
        self.drug_toxicity_cache[cache_key] = toxicity_genes
        
        return toxicity_genes
    
    def analyze_protective_effects(self, protective_drug_id, toxicity_drug_id, toxicity_terms=None):
        """
        分析保护性药物对毒性药物的保护作用
        
        Parameters:
        - protective_drug_id: 保护性药物ID
        - toxicity_drug_id: 毒性药物ID
        - toxicity_terms: 毒性相关术语列表（可选）
        
        Returns:
        - 保护作用分析结果
        """
        # 检查缓存
        cache_key = (protective_drug_id, toxicity_drug_id, tuple(toxicity_terms) if toxicity_terms else None)
        if cache_key in self.protection_cache:
            return self.protection_cache[cache_key]
        
        # 识别毒性药物的毒性相关基因
        toxicity_genes = self.identify_toxicity_genes(toxicity_drug_id, toxicity_terms)
        
        if not toxicity_genes:
            logger.warning(f"未找到药物 {toxicity_drug_id} 的毒性相关基因")
            return []
        
        # 分析保护性药物对毒性基因的调节作用
        protective_effects = []
        
        for gene_info in toxicity_genes:
            gene_id = gene_info["gene_id"]
            
            # 检查保护性药物是否调节该基因
            if self.graph.has_edge(protective_drug_id, gene_id):
                protective_direction = self.graph[protective_drug_id][gene_id].get('direction', 0)
                protective_confidence = self.graph[protective_drug_id][gene_id].get('confidence', 0.5)
            else:
                # 尝试找到间接调节关系
                indirect_relation = self.psr_engine.infer_indirect_relations(protective_drug_id, gene_id, 2)
                protective_direction = indirect_relation["direction"]
                protective_confidence = indirect_relation["probability"]
                
                if protective_direction is None or protective_confidence == 0:
                    continue
            
            # 计算保护效果
            # 如果毒性药物上调一个促进毒性的基因，而保护性药物下调该基因，则有保护作用
            # 如果毒性药物下调一个抑制毒性的基因，而保护性药物上调该基因，则有保护作用
            toxicity_effect = gene_info["drug_gene_direction"] * gene_info["gene_toxicity_direction"]
            protective_effect = -1 * protective_direction * gene_info["gene_toxicity_direction"]
            
            # 只有当保护效果为正时，才有保护作用
            if protective_effect > 0:
                protective_effects.append({
                    "gene_id": gene_id,
                    "gene_name": gene_info["gene_name"],
                    "toxicity_name": gene_info["toxicity_name"],
                    "toxicity_drug_effect": gene_info["drug_gene_direction"],
                    "protective_drug_effect": protective_direction,
                    "gene_toxicity_relation": gene_info["gene_toxicity_direction"],
                    "protection_score": protective_effect * protective_confidence,
                    "confidence": protective_confidence,
                    "is_direct": self.graph.has_edge(protective_drug_id, gene_id)
                })
        
        # 排序
        protective_effects.sort(key=lambda x: x["protection_score"], reverse=True)
        
        # 缓存结果
        self.protection_cache[cache_key] = protective_effects
        
        return protective_effects
    
    def calculate_toxicity_reduction_score(self, protective_drug_id, toxicity_drug_id, toxicity_terms=None):
        """
        计算毒性减轻评分
        
        Parameters:
        - protective_drug_id: 保护性药物ID
        - toxicity_drug_id: 毒性药物ID
        - toxicity_terms: 毒性相关术语列表（可选）
        
        Returns:
        - 毒性减轻评分和详细信息
        """
        # 分析保护作用
        protective_effects = self.analyze_protective_effects(protective_drug_id, toxicity_drug_id, toxicity_terms)
        
        if not protective_effects:
            return {"score": 0.0, "mechanisms": [], "summary": {}}
        
        # 计算总体保护评分
        total_score = sum(effect["protection_score"] for effect in protective_effects) / len(protective_effects)
        
        # 按毒性类型分类
        toxicity_types = {}
        for effect in protective_effects:
            toxicity_name = effect["toxicity_name"]
            if toxicity_name not in toxicity_types:
                toxicity_types[toxicity_name] = []
            toxicity_types[toxicity_name].append(effect)
        
        # 准备摘要信息
        summary = {
            "total_protective_genes": len(protective_effects),
            "toxicity_types": len(toxicity_types),
            "direct_mechanisms": sum(1 for effect in protective_effects if effect["is_direct"]),
            "indirect_mechanisms": sum(1 for effect in protective_effects if not effect["is_direct"])
        }
        
        return {
            "score": total_score,
            "mechanisms": protective_effects,
            "toxicity_types": toxicity_types,
            "summary": summary
        }
    
    def rank_protective_candidates(self, toxicity_drug_id, candidate_drugs, toxicity_terms=None):
        """
        对保护性药物候选进行排序
        
        Parameters:
        - toxicity_drug_id: 毒性药物ID
        - candidate_drugs: 候选药物ID列表
        - toxicity_terms: 毒性相关术语列表（可选）
        
        Returns:
        - 按保护评分排序的候选药物列表
        """
        rankings = []
        
        for candidate_id in tqdm(candidate_drugs, desc="评估保护性候选药物"):
            # 计算保护评分
            protection_result = self.calculate_toxicity_reduction_score(candidate_id, toxicity_drug_id, toxicity_terms)
            
            # 添加到排名列表
            rankings.append({
                "drug_id": candidate_id,
                "drug_name": self.graph.nodes[candidate_id].get("name", "") if self.graph else "",
                "protection_score": protection_result["score"],
                "protective_gene_count": protection_result["summary"]["total_protective_genes"],
                "toxicity_type_count": protection_result["summary"]["toxicity_types"]
            })
        
        # 按评分排序
        rankings.sort(key=lambda x: x["protection_score"], reverse=True)
        
        return rankings 
