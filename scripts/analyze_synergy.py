#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
药物协同分析脚本
分析两种药物之间的协同作用机制
"""

import os
import sys
import argparse
import logging
import pandas as pd
import networkx as nx
import json
from datetime import datetime
from core.graph_builder import create_drug_synergy_subgraph
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from config.settings import PATHS
from core.graph_builder import build_networkx_graph
from analysis.psr_engine import PSREngine
from analysis.synergy_analyzer import SynergyAnalyzer
from analysis.pathway_analyzer import PathwayAnalyzer
from utils.file_utils import load_csv, save_to_json, save_to_csv
from utils.visualization import generate_network_visualization, generate_mechanism_diagram

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='iKraph Drug Synergy Analysis Tool')
    
    # 输入输出参数
    parser.add_argument('--entities_file', type=str, default=os.path.join(PATHS['tables_dir'], 'all_entities.csv'), help='实体CSV文件路径')
    parser.add_argument('--relations_file', type=str, default=os.path.join(PATHS['tables_dir'], 'relations.csv'), help='关系CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=PATHS['output_dir'], help='输出目录路径')
    
    # 药物和疾病参数
    parser.add_argument('--drug1', type=str, required=True, help='第一种药物名称')
    parser.add_argument('--drug2', type=str, required=True, help='第二种药物名称')
    parser.add_argument('--disease', type=str, required=True, help='疾病名称')
    parser.add_argument('--exact_match', action='store_true', help='使用精确匹配（默认为部分匹配）')
    
    # 分析参数
    parser.add_argument('--max_path_length', type=int, default=2, help='最大路径长度')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='最小置信度')
    
    # 输出参数
    parser.add_argument('--report_format', type=str, choices=['json', 'csv', 'both'], default='both', help='报告格式')
    parser.add_argument('--generate_viz', action='store_true', default=True, help='生成可视化')
    
    return parser.parse_args()

def find_entity_by_name(entities_df, name, exact_match=False):
    """根据名称查找实体"""
    name = name.lower()
    
    if exact_match:
        # 精确匹配
        matches = entities_df[
            (entities_df['Name'].str.lower() == name) |
            (entities_df['Official_Name'].str.lower() == name) |
            (entities_df['Common_Name'].str.lower() == name)
        ]
    else:
        # 部分匹配
        matches = entities_df[
            entities_df['Name'].str.lower().str.contains(name, na=False) |
            entities_df['Official_Name'].str.lower().str.contains(name, na=False) |
            entities_df['Common_Name'].str.lower().str.contains(name, na=False)
        ]
    
    if len(matches) == 0:
        logger.warning(f"未找到匹配'{name}'的实体")
        return None
    
    # 返回最匹配的实体（假设是第一个）
    return matches.iloc[0]

def run_analysis(args):
    """运行协同分析流程"""
    logger.info("启动药物协同分析流程...")
    
    # 加载实体和关系数据
    entities_df = load_csv(args.entities_file)
    relations_df = load_csv(args.relations_file)
    
    if entities_df is None or relations_df is None:
        logger.error("加载数据失败")
        return False
    
    # 查找药物和疾病实体
    drug1_entity = find_entity_by_name(entities_df, args.drug1, args.exact_match)
    drug2_entity = find_entity_by_name(entities_df, args.drug2, args.exact_match)
    disease_entity = find_entity_by_name(entities_df, args.disease, args.exact_match)
    
    if drug1_entity is None or drug2_entity is None or disease_entity is None:
        logger.error("未找到指定的药物或疾病实体")
        return False
    
    drug1_id = str(drug1_entity['ID'])
    drug2_id = str(drug2_entity['ID'])
    disease_id = str(disease_entity['ID'])
    
    logger.info(f"分析 {drug1_entity['Name']} (ID: {drug1_id}) 和 {drug2_entity['Name']} (ID: {drug2_id}) 对 {disease_entity['Name']} (ID: {disease_id}) 的协同作用")
    
    # 构建网络图
    G = build_networkx_graph(entities_df, relations_df)
    logger.info(f"构建了包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边的网络图")
    
    # 创建分析器
    psr_engine = PSREngine(G)
    synergy_analyzer = SynergyAnalyzer(G, psr_engine)
    pathway_analyzer = PathwayAnalyzer(G)
    
    # 执行协同分析
    logger.info("执行协同分析...")
    synergy_result = synergy_analyzer.calculate_synergy_score(
        drug1_id, drug2_id, disease_id, args.max_path_length)
    
    # 检查结果
    if not synergy_result or synergy_result["score"] == 0:
        logger.warning("未发现有效的协同作用机制")
    else:
        logger.info(f"协同评分: {synergy_result['score']:.4f}")
        logger.info(f"发现 {len(synergy_result['common_targets'])} 个共同靶点")
        logger.info(f"发现 {len(synergy_result['complementary_mechanisms'])} 个互补机制")
    
    # 准备输出目录
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 准备输出文件名
    base_filename = f"{args.drug1.replace(' ', '_')}_{args.drug2.replace(' ', '_')}_{args.disease.replace(' ', '_')}"
    
    # 保存结果
    if args.report_format in ['json', 'both']:
        json_file = os.path.join(reports_dir, f"{base_filename}_synergy.json")
        save_to_json(synergy_result, json_file)
    
    if args.report_format in ['csv', 'both']:
        # 保存共同靶点
        if synergy_result['common_targets']:
            common_targets_file = os.path.join(reports_dir, f"{base_filename}_common_targets.csv")
            save_to_csv(synergy_result['common_targets'], common_targets_file)
        
        # 保存互补机制
        if synergy_result['complementary_mechanisms']:
            complementary_mechanisms_file = os.path.join(reports_dir, f"{base_filename}_complementary_mechanisms.csv")
            save_to_csv(synergy_result['complementary_mechanisms'], complementary_mechanisms_file)
    
    # 生成可视化
    if args.generate_viz:
        graphs_dir = os.path.join(args.output_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        # 生成网络可视化
        viz_file = os.path.join(graphs_dir, f"{base_filename}_network.html")
        
        # 获取完整的实体名称
        drug1_name = drug1_entity['Name']
        drug2_name = drug2_entity['Name']
        disease_name = disease_entity['Name']
        
        # 创建更有意义的标题
        viz_title = f"药物协同网络: {drug1_name} + {drug2_name} → {disease_name}"
        
        # 添加药物靶点和相关节点
        subgraph = create_drug_synergy_subgraph(G, drug1_id, drug2_id, disease_id, max_intermediate_nodes=50)
        
        # 调用可视化函数
        generate_network_visualization(subgraph, viz_file, title=viz_title)
        
        # 生成机制图
        if synergy_result['common_targets'] or synergy_result['complementary_mechanisms']:
            mechanism_file = os.path.join(graphs_dir, f"{base_filename}_mechanism.png")
            
            # 准备机制数据
            mechanisms = []
            
            # 添加共同靶点
            for target in synergy_result['common_targets']:
                mechanisms.append({
                    'drug_id': drug1_id,
                    'drug_name': drug1_name,
                    'gene_id': target['target_id'],
                    'gene_name': target['target_name'],
                    'target_id': disease_id,
                    'target_name': disease_name,
                    'drug_gene_direction': target['drug1_regulation'],
                    'gene_target_direction': target['disease_direction'],
                    'mechanism_type': 'common_target'
                })
                
                mechanisms.append({
                    'drug_id': drug2_id,
                    'drug_name': drug2_name,
                    'gene_id': target['target_id'],
                    'gene_name': target['target_name'],
                    'target_id': disease_id,
                    'target_name': disease_name,
                    'drug_gene_direction': target['drug2_regulation'],
                    'gene_target_direction': target['disease_direction'],
                    'mechanism_type': 'common_target'
                })
            
            # 添加互补机制
            for mechanism in synergy_result['complementary_mechanisms']:
                mechanisms.append({
                    'drug_id': drug1_id,
                    'drug_name': drug1_name,
                    'gene_id': mechanism['target1_id'],
                    'gene_name': mechanism['target1_name'],
                    'target_id': disease_id,
                    'target_name': disease_name,
                    'drug_gene_direction': mechanism['drug1_regulation'],
                    'gene_target_direction': 0,  # 未直接连接到疾病
                    'mechanism_type': 'complementary'
                })
                
                mechanisms.append({
                    'drug_id': drug2_id,
                    'drug_name': drug2_name,
                    'gene_id': mechanism['target2_id'],
                    'gene_name': mechanism['target2_name'],
                    'target_id': disease_id,
                    'target_name': disease_name,
                    'drug_gene_direction': mechanism['drug2_regulation'],
                    'gene_target_direction': 0,  # 未直接连接到疾病
                    'mechanism_type': 'complementary'
                })
            
            generate_mechanism_diagram(mechanisms, mechanism_file, 
                                     title=f"协同作用机制: {drug1_name} + {drug2_name} → {disease_name}")
    
    logger.info("药物协同分析流程完成")
    return True
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
                confidence = G[drug1_id][target].get('confidence', G[drug1_id][target].get('Confidence', 0))
                drug1_targets.append((target, confidence))
        
        drug1_targets.sort(key=lambda x: x[1], reverse=True)
        
        # 按关系强度排序药物2的靶点
        drug2_targets = []
        for target in drug2_neighbors:
            if target in G[drug2_id]:
                confidence = G[drug2_id][target].get('confidence', G[drug2_id][target].get('Confidence', 0))
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
        import random
        selected_others = random.sample(other_nodes, max_intermediate_nodes)
        nodes_to_keep = set(essential_nodes).union(selected_others)
    
    # 创建子图
    subgraph = G.subgraph(nodes_to_keep)
    
    # 确保所有节点都有Name或name属性
    for node_id in subgraph.nodes():
        node_attrs = subgraph.nodes[node_id]
        if 'Name' not in node_attrs and 'name' not in node_attrs:
            # 如果没有名称，添加一个基于ID的名称
            if node_id == drug1_id:
                subgraph.nodes[node_id]['name'] = "Drug 1"
            elif node_id == drug2_id:
                subgraph.nodes[node_id]['name'] = "Drug 2"
            elif node_id == disease_id:
                subgraph.nodes[node_id]['name'] = "Disease"
            else:
                subgraph.nodes[node_id]['name'] = f"Node {node_id}"
    
    logger.info(f"创建的药物协同子图包含 {len(subgraph.nodes)} 个节点和 {len(subgraph.edges)} 条边")
    
    return subgraph
def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('synergy_analysis.log', mode='w', encoding='utf-8')
        ]
    )
    
    # 解析参数
    args = parse_arguments()
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"开始处理时间: {start_time}")
    
    # 运行分析流程
    success = run_analysis(args)
    
    # 记录结束时间
    end_time = datetime.now()
    processing_time = end_time - start_time
    logger.info(f"结束处理时间: {end_time}")
    logger.info(f"总处理时间: {processing_time}")
    
    # 返回状态码
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 


