#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
毒性减轻分析脚本
分析一种药物如何减轻另一种药物的毒性
"""

import os
import sys
import argparse
import logging
import pandas as pd
import networkx as nx
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from config.settings import PATHS
from core.graph_builder import build_networkx_graph
from analysis.psr_engine import PSREngine
from analysis.toxicity_reducer import ToxicityReducer
from utils.file_utils import load_csv, save_to_json, save_to_csv
from utils.visualization import generate_network_visualization, generate_mechanism_diagram

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='iKraph Drug Toxicity Reduction Analysis Tool')
    
    # 输入输出参数
    parser.add_argument('--entities_file', type=str, default=os.path.join(PATHS['tables_dir'], 'all_entities.csv'), help='实体CSV文件路径')
    parser.add_argument('--relations_file', type=str, default=os.path.join(PATHS['tables_dir'], 'relations.csv'), help='关系CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=PATHS['output_dir'], help='输出目录路径')
    
    # 药物参数
    parser.add_argument('--toxic_drug', type=str, required=True, help='具有毒性的药物名称')
    parser.add_argument('--protective_drug', type=str, required=True, help='可能具有保护作用的药物名称')
    parser.add_argument('--exact_match', action='store_true', help='使用精确匹配（默认为部分匹配）')
    
    # 分析参数
    parser.add_argument('--toxicity_terms', type=str, nargs='+', default=['toxicity', 'adverse', 'damage', 'injury', 'side effect'], help='毒性相关术语')
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
    """运行毒性减轻分析流程"""
    logger.info("启动药物毒性减轻分析流程...")
    
    # 加载实体和关系数据
    entities_df = load_csv(args.entities_file)
    relations_df = load_csv(args.relations_file)
    
    if entities_df is None or relations_df is None:
        logger.error("加载数据失败")
        return False
    
    # 查找药物实体
    toxic_drug_entity = find_entity_by_name(entities_df, args.toxic_drug, args.exact_match)
    protective_drug_entity = find_entity_by_name(entities_df, args.protective_drug, args.exact_match)
    
    if toxic_drug_entity is None or protective_drug_entity is None:
        logger.error("未找到指定的药物实体")
        return False
    
    toxic_drug_id = str(toxic_drug_entity['ID'])
    protective_drug_id = str(protective_drug_entity['ID'])
    
    logger.info(f"分析 {protective_drug_entity['Name']} (ID: {protective_drug_id}) 对 {toxic_drug_entity['Name']} (ID: {toxic_drug_id}) 毒性的减轻作用")
    
    # 构建网络图
    G = build_networkx_graph(entities_df, relations_df)
    logger.info(f"构建了包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边的网络图")
    
    # 创建分析器
    psr_engine = PSREngine(G)
    toxicity_reducer = ToxicityReducer(G, psr_engine)
    
    # 执行毒性减轻分析
    logger.info("执行毒性减轻分析...")
    toxicity_result = toxicity_reducer.calculate_toxicity_reduction_score(
        protective_drug_id, toxic_drug_id, args.toxicity_terms)
    
    # 检查结果
    if not toxicity_result or toxicity_result["score"] == 0:
        logger.warning("未发现有效的毒性减轻机制")
    else:
        logger.info(f"毒性减轻评分: {toxicity_result['score']:.4f}")
        logger.info(f"发现 {len(toxicity_result['mechanisms'])} 个保护机制")
        logger.info(f"影响 {len(toxicity_result['toxicity_types'])} 种毒性类型")
    
    # 准备输出目录
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 准备输出文件名
    base_filename = f"{args.protective_drug.replace(' ', '_')}_{args.toxic_drug.replace(' ', '_')}_toxicity_reduction"
    
    # 保存结果
    if args.report_format in ['json', 'both']:
        json_file = os.path.join(reports_dir, f"{base_filename}.json")
        save_to_json(toxicity_result, json_file)
    
    if args.report_format in ['csv', 'both'] and toxicity_result['mechanisms']:
        # 保存保护机制
        mechanisms_file = os.path.join(reports_dir, f"{base_filename}_mechanisms.csv")
        save_to_csv(toxicity_result['mechanisms'], mechanisms_file)
        
        # 保存毒性类型
        toxicity_types = []
        for toxicity_name, mechanisms in toxicity_result['toxicity_types'].items():
            for mechanism in mechanisms:
                mechanism['toxicity_type'] = toxicity_name
                toxicity_types.append(mechanism)
        
        if toxicity_types:
            toxicity_types_file = os.path.join(reports_dir, f"{base_filename}_toxicity_types.csv")
            save_to_csv(toxicity_types, toxicity_types_file)
    
    # 生成可视化
    if args.generate_viz and toxicity_result['mechanisms']:
        graphs_dir = os.path.join(args.output_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        # 生成机制图
        mechanism_file = os.path.join(graphs_dir, f"{base_filename}_mechanism.png")
        
        # 使用实体名称创建标题
        toxic_drug_name = toxic_drug_entity['Name']
        protective_drug_name = protective_drug_entity['Name']
        
        mechanism_title = f"毒性减轻机制: {protective_drug_name} 保护 {toxic_drug_name} 毒性"
        
        generate_mechanism_diagram(toxicity_result['mechanisms'], mechanism_file, title=mechanism_title)
        
        # 收集毒性节点和基因节点
        nodes_to_keep = set([toxic_drug_id, protective_drug_id])
        for mechanism in toxicity_result['mechanisms']:
            nodes_to_keep.add(mechanism['gene_id'])
            if 'toxicity_node' in mechanism:
                nodes_to_keep.add(mechanism['toxicity_node'])
        
        # 创建子图
        subgraph = G.subgraph(nodes_to_keep)
        
        # 生成网络可视化
        viz_file = os.path.join(graphs_dir, f"{base_filename}_network.html")
        
        network_title = f"毒性减轻网络: {protective_drug_name} 保护 {toxic_drug_name} 毒性"
        
        generate_network_visualization(subgraph, viz_file, title=network_title)
    
    logger.info("药物毒性减轻分析流程完成")
    return True

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('toxicity_analysis.log', mode='w', encoding='utf-8')
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
