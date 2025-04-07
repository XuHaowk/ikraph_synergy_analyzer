#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通路分析模块
分析生物通路富集和药物作用
"""

import logging
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Union, Optional

logger = logging.getLogger(__name__)

class PathwayAnalyzer:
    """生物通路分析器"""
    
    def __init__(self, graph=None):
        """
        初始化通路分析器
        
        Parameters:
        - graph: NetworkX图对象（可选）
        """
        self.graph = graph
        
        # 缓存
        self.pathway_gene_cache = {}
        self.gene_pathway_cache = {}
    
    def set_graph(self, graph):
        """设置图对象"""
        self.graph = graph
        # 清除缓存
        self.pathway_gene_cache = {}
        self.gene_pathway_cache = {}
    
    def get_pathway_genes(self, pathway_id):
        """
        获取通路中的基因
        
        Parameters:
        - pathway_id: 通路ID
        
        Returns:
        - 基因ID列表
        """
        # 检查缓存
        if pathway_id in self.pathway_gene_cache:
            return self.pathway_gene_cache[pathway_id]
        
        if self.graph is None:
            logger.error("图对象未设置")
            return []
        
        # 检查通路节点是否存在
        if pathway_id not in self.graph:
            logger.warning(f"通路 {pathway_id} 不在图中")
            return []
        
        # 获取通路相关的基因
        pathway_genes = []
        for gene_id in self.graph.neighbors(pathway_id):
            if self.graph.nodes[gene_id].get('type') == "Gene":
                pathway_genes.append(gene_id)
        
        # 缓存结果
        self.pathway_gene_cache[pathway_id] = pathway_genes
        
        return pathway_genes
    
    def get_gene_pathways(self, gene_id):
        """
        获取基因参与的通路
        
        Parameters:
        - gene_id: 基因ID
        
        Returns:
        - 通路ID列表
        """
        # 检查缓存
        if gene_id in self.gene_pathway_cache:
            return self.gene_pathway_cache[gene_id]
        
        if self.graph is None:
            logger.error("图对象未设置")
            return []
        
        # 检查基因节点是否存在
        if gene_id not in self.graph:
            logger.warning(f"基因 {gene_id} 不在图中")
            return []
        
        # 获取基因相关的通路
        gene_pathways = []
        for pathway_id in self.graph.neighbors(gene_id):
            if self.graph.nodes[pathway_id].get('type') == "Pathway":
                gene_pathways.append(pathway_id)
        
        # 缓存结果
        self.gene_pathway_cache[gene_id] = gene_pathways
        
        return gene_pathways
    
    def get_all_pathways(self):
        """
        获取图中的所有通路
        
        Returns:
        - 通路ID列表
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return []
        
        # 查找所有通路类型的节点
        pathways = [node for node, attrs in self.graph.nodes(data=True) 
                   if attrs.get('type') == "Pathway"]
        
        return pathways
    
    def perform_pathway_enrichment(self, gene_list, background_genes=None, p_value_cutoff=0.05):
        """
        执行通路富集分析
        
        Parameters:
        - gene_list: 目标基因列表
        - background_genes: 背景基因列表（可选）
        - p_value_cutoff: P值阈值
        
        Returns:
        - 富集通路列表，包含统计信息
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return []
        
        # 获取所有通路
        all_pathways = self.get_all_pathways()
        
        # 如果未提供背景基因，使用图中所有基因
        if background_genes is None:
            background_genes = [node for node, attrs in self.graph.nodes(data=True) 
                              if attrs.get('type') == "Gene"]
        
        # 将基因列表转换为集合
        gene_set = set(gene_list)
        background_set = set(background_genes)
        
        # 总背景基因数
        N = len(background_set)
        
        # 目标基因数
        n = len(gene_set)
        
        # 富集结果
        enrichment_results = []
        
        for pathway_id in tqdm(all_pathways, desc="分析通路富集"):
            # 获取通路基因
            pathway_genes = set(self.get_pathway_genes(pathway_id))
            
            # 通路基因数
            K = len(pathway_genes)
            
            if K == 0:
                continue
            
            # 与背景基因集的交集
            pathway_background = pathway_genes.intersection(background_set)
            
            # 调整通路基因数
            K = len(pathway_background)
            
            # 与目标基因集的交集
            pathway_hits = pathway_genes.intersection(gene_set)
            
            # 命中基因数
            k = len(pathway_hits)
            
            if k == 0:
                continue
            
            # 计算超几何分布的P值
            # P(X >= k)
            p_value = stats.hypergeom.sf(k-1, N, K, n)
            
            # 如果P值小于阈值，添加到结果
            if p_value <= p_value_cutoff:
                # 计算富集倍数
                fold_enrichment = (k/n) / (K/N)
                
                enrichment_results.append({
                    "pathway_id": pathway_id,
                    "pathway_name": self.graph.nodes[pathway_id].get('name', ''),
                    "gene_count": k,
                    "pathway_size": K,
                    "background_size": N,
                    "p_value": p_value,
                    "fold_enrichment": fold_enrichment,
                    "genes": list(pathway_hits)
                })
        
        # 按P值排序
        enrichment_results.sort(key=lambda x: x['p_value'])
        
        return enrichment_results
    
    def analyze_pathway_overlap(self, pathway_id1, pathway_id2):
        """
        分析两个通路的基因重叠程度
        
        Parameters:
        - pathway_id1: 第一个通路ID
        - pathway_id2: 第二个通路ID
        
        Returns:
        - 重叠分析结果
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return {"overlap_coefficient": 0, "jaccard_index": 0, "genes": []}
        
        # 获取两个通路的基因
        genes1 = set(self.get_pathway_genes(pathway_id1))
        genes2 = set(self.get_pathway_genes(pathway_id2))
        
        if not genes1 or not genes2:
            return {"overlap_coefficient": 0, "jaccard_index": 0, "genes": []}
        
        # 计算交集
        intersection = genes1.intersection(genes2)
        
        # 计算Overlap系数: |A∩B| / min(|A|,|B|)
        overlap_coefficient = len(intersection) / min(len(genes1), len(genes2))
        
        # 计算Jaccard指数: |A∩B| / |A∪B|
        jaccard_index = len(intersection) / len(genes1.union(genes2))
        
        return {
            "pathway1_id": pathway_id1,
            "pathway1_name": self.graph.nodes[pathway_id1].get('name', ''),
            "pathway1_size": len(genes1),
            "pathway2_id": pathway_id2,
            "pathway2_name": self.graph.nodes[pathway_id2].get('name', ''),
            "pathway2_size": len(genes2),
            "overlap_coefficient": overlap_coefficient,
            "jaccard_index": jaccard_index,
            "shared_gene_count": len(intersection),
            "shared_genes": list(intersection)
        }
    
    def analyze_drug_pathway_impact(self, drug_id, direct_only=False):
        """
        分析药物对通路的影响
        
        Parameters:
        - drug_id: 药物ID
        - direct_only: 是否仅分析直接影响
        
        Returns:
        - 通路影响分析结果
        """
        if self.graph is None:
            logger.error("图对象未设置")
            return []
        
        # 药物靶点
        drug_targets = []
        for target_id in self.graph.neighbors(drug_id):
            if self.graph.nodes[target_id].get('type') == "Gene":
                drug_targets.append(target_id)
        
        # 收集受影响的通路
        affected_pathways = {}
        
        # 分析每个靶点
        for target_id in drug_targets:
            # 获取靶点的通路
            target_pathways = self.get_gene_pathways(target_id)
            
            # 获取药物对靶点的调节方向
            if self.graph.has_edge(drug_id, target_id):
                direction = self.graph[drug_id][target_id].get('direction', 0)
                confidence = self.graph[drug_id][target_id].get('confidence', 0.5)
            else:
                direction = 0
                confidence = 0
            
            # 更新通路影响
            for pathway_id in target_pathways:
                if pathway_id not in affected_pathways:
                    affected_pathways[pathway_id] = {
                        "pathway_id": pathway_id,
                        "pathway_name": self.graph.nodes[pathway_id].get('name', ''),
                        "affected_genes": [],
                        "total_impact": 0,
                        "confidence": 0
                    }
                
                # 添加受影响的基因
                affected_pathways[pathway_id]["affected_genes"].append({
                    "gene_id": target_id,
                    "gene_name": self.graph.nodes[target_id].get('name', ''),
                    "direction": direction,
                    "confidence": confidence
                })
                
                # 更新总体影响
                affected_pathways[pathway_id]["total_impact"] += direction * confidence
                affected_pathways[pathway_id]["confidence"] = max(affected_pathways[pathway_id]["confidence"], confidence)
        
        # 如果不仅限于直接影响，分析间接影响
        if not direct_only:
            # 获取药物靶点的下游基因
            downstream_genes = set()
            for target_id in drug_targets:
                for downstream_id in self.graph.neighbors(target_id):
                    if self.graph.nodes[downstream_id].get('type') == "Gene" and downstream_id not in drug_targets:
                        downstream_genes.add(downstream_id)
            
            # 分析下游基因的通路
            for gene_id in downstream_genes:
                # 获取基因的通路
                gene_pathways = self.get_gene_pathways(gene_id)
                
                # 更新通路影响
                for pathway_id in gene_pathways:
                    if pathway_id not in affected_pathways:
                        affected_pathways[pathway_id] = {
                            "pathway_id": pathway_id,
                            "pathway_name": self.graph.nodes[pathway_id].get('name', ''),
                            "affected_genes": [],
                            "total_impact": 0,
                            "confidence": 0
                        }
                    
                    # 对于间接影响，影响较小
                    affected_pathways[pathway_id]["affected_genes"].append({
                        "gene_id": gene_id,
                        "gene_name": self.graph.nodes[gene_id].get('name', ''),
                        "direction": 0,  # 方向未知
                        "confidence": 0.3,  # 较低的置信度
                        "is_indirect": True
                    })
        
        # 转换为列表并排序
        result = list(affected_pathways.values())
        result.sort(key=lambda x: abs(x['total_impact']), reverse=True)
        
        return result 
