#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
肝肾毒性减轻分析脚本
分析黄芩苷如何减轻汉防己甲素的肝肾毒性
"""

import os
import sys
import argparse
import logging
import pandas as pd
import networkx as nx
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from config.settings import PATHS
from core.graph_builder import build_networkx_graph
from analysis.psr_engine import PSREngine
from analysis.toxicity_reducer import ToxicityReducer
from utils.file_utils import load_csv, save_to_json, save_to_csv
from utils.visualization import generate_network_visualization, generate_mechanism_diagram

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hepatorenal_toxicity_analysis.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='肝肾毒性减轻分析工具')
    
    # 输入输出参数
    parser.add_argument('--entities_file', type=str, default=os.path.join(PATHS['tables_dir'], 'all_entities.csv'), help='实体CSV文件路径')
    parser.add_argument('--relations_file', type=str, default=os.path.join(PATHS['tables_dir'], 'relations.csv'), help='关系CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=PATHS['output_dir'], help='输出目录路径')
    
    # 药物参数
    parser.add_argument('--toxic_drug', type=str, default="Tetrandrine", help='具有毒性的药物名称')
    parser.add_argument('--protective_drug', type=str, default="Baicalin", help='可能具有保护作用的药物名称')
    parser.add_argument('--exact_match', action='store_true', help='使用精确匹配（默认为部分匹配）')
    
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

def run_hepatorenal_toxicity_analysis(args):
    """运行肝肾毒性减轻分析流程"""
    logger.info("启动肝肾毒性减轻分析流程...")
    
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
    
    logger.info(f"分析 {protective_drug_entity['Name']} (ID: {protective_drug_id}) 对 {toxic_drug_entity['Name']} (ID: {toxic_drug_id}) 肝肾毒性的减轻作用")
    
    # 构建网络图
    G = build_networkx_graph(entities_df, relations_df)
    logger.info(f"构建了包含 {len(G.nodes)} 个节点和 {len(G.edges)} 条边的网络图")
    
    # 创建分析器
    psr_engine = PSREngine(G)
    toxicity_reducer = ToxicityReducer(G, psr_engine)
    
    # 定义肝肾毒性术语
    hepatorenal_toxicity_terms = [
        # 肝毒性相关术语
        "hepatotoxicity", "liver damage", "liver injury", "liver toxicity", 
        "hepatic injury", "hepatic damage", "hepatic failure", "hepatitis",
        "肝毒性", "肝损伤", "肝损害", "肝衰竭", "肝炎",
        
        # 肾毒性相关术语
        "nephrotoxicity", "kidney damage", "kidney injury", "kidney toxicity",
        "renal injury", "renal damage", "renal failure", "nephritis",
        "肾毒性", "肾损伤", "肾损害", "肾衰竭", "肾炎"
    ]
    
    # 执行肝肾毒性减轻分析
    logger.info("执行肝肾毒性减轻分析...")
    toxicity_result = toxicity_reducer.calculate_toxicity_reduction_score(
        protective_drug_id, toxic_drug_id, hepatorenal_toxicity_terms
    )
    
    # 检查结果
    if not toxicity_result or toxicity_result["score"] == 0:
        logger.warning("未发现有效的肝肾毒性减轻机制")
    else:
        logger.info(f"毒性减轻评分: {toxicity_result['score']:.4f}")
        logger.info(f"发现 {len(toxicity_result['mechanisms'])} 个保护机制")
        
        # 按保护作用类型分类
        liver_mechanisms = []
        kidney_mechanisms = []
        other_mechanisms = []
        
        for mechanism in toxicity_result["mechanisms"]:
            toxicity_name = mechanism.get("toxicity_name", "").lower()
            
            if any(term in toxicity_name for term in ["liver", "hepat", "肝"]):
                liver_mechanisms.append(mechanism)
            elif any(term in toxicity_name for term in ["kidney", "renal", "nephr", "肾"]):
                kidney_mechanisms.append(mechanism)
            else:
                other_mechanisms.append(mechanism)
        
        logger.info(f"肝脏保护机制: {len(liver_mechanisms)}")
        logger.info(f"肾脏保护机制: {len(kidney_mechanisms)}")
        logger.info(f"其他保护机制: {len(other_mechanisms)}")
    
    # 准备输出目录
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 准备输出文件名
    base_filename = f"{args.protective_drug.replace(' ', '_')}_{args.toxic_drug.replace(' ', '_')}_hepatorenal_toxicity_reduction"
    
    # 保存结果
    if args.report_format in ['json', 'both']:
        json_file = os.path.join(reports_dir, f"{base_filename}.json")
        save_to_json(toxicity_result, json_file)
    
    if args.report_format in ['csv', 'both'] and toxicity_result['mechanisms']:
        # 保存保护机制
        mechanisms_file = os.path.join(reports_dir, f"{base_filename}_mechanisms.csv")
        save_to_csv(toxicity_result['mechanisms'], mechanisms_file)
        
        # 分别保存肝脏和肾脏保护机制
        if liver_mechanisms:
            liver_mechanisms_file = os.path.join(reports_dir, f"{base_filename}_liver_mechanisms.csv")
            save_to_csv(liver_mechanisms, liver_mechanisms_file)
        
        if kidney_mechanisms:
            kidney_mechanisms_file = os.path.join(reports_dir, f"{base_filename}_kidney_mechanisms.csv")
            save_to_csv(kidney_mechanisms, kidney_mechanisms_file)
    
    # 生成机制图表
    if args.generate_viz and toxicity_result['mechanisms']:
        graphs_dir = os.path.join(args.output_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        # 创建机制总结图
        if toxicity_result["score"] > 0:
            # 创建分类图表
            plt.figure(figsize=(10, 6))
            
            # 设置数据
            categories = ['肝脏保护', '肾脏保护', '其他保护']
            values = [len(liver_mechanisms), len(kidney_mechanisms), len(other_mechanisms)]
            
            # 计算百分比
            total = sum(values)
            percentages = [100 * v / total if total > 0 else 0 for v in values]
            
            # 绘制条形图
            bars = plt.bar(categories, values, color=['#1f77b4', '#2ca02c', '#7f7f7f'])
            
            # 添加标签
            for bar, percentage in zip(bars, percentages):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{percentage:.1f}%', ha='center', va='bottom')
            
            plt.title(f'{args.protective_drug} 对 {args.toxic_drug} 肝肾毒性的保护机制分布')
            plt.ylabel('保护机制数量')
            plt.tight_layout()
            
            # 保存图表
            summary_chart_file = os.path.join(graphs_dir, f"{base_filename}_summary.png")
            plt.savefig(summary_chart_file, dpi=300)
            plt.close()
            
        # 生成机制图
        mechanism_file = os.path.join(graphs_dir, f"{base_filename}_mechanism.png")
        generate_mechanism_diagram(toxicity_result['mechanisms'], mechanism_file,
                                  title=f"肝肾毒性减轻机制: {protective_drug_entity['Name']} 保护 {toxic_drug_entity['Name']} 肝肾毒性")
        
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
        generate_network_visualization(subgraph, viz_file,
                                     title=f"肝肾毒性减轻网络: {protective_drug_entity['Name']} 保护 {toxic_drug_entity['Name']} 肝肾毒性")
    
    # 生成肝肾毒性减轻评估报告
    report_text = f"""
    # {args.protective_drug} 对 {args.toxic_drug} 肝肾毒性的减轻效应分析报告
    
    ## 概述
    
    本报告利用iKraph知识图谱分析 {args.protective_drug} 对 {args.toxic_drug} 的肝肾毒性减轻作用。
    
    ## 总体保护评分
    
    保护评分: {toxicity_result["score"]:.4f} (范围: 0-1, 越高表示保护作用越显著)
    
    ## 保护机制分布
    
    - 肝脏保护机制: {len(liver_mechanisms)} 个
    - 肾脏保护机制: {len(kidney_mechanisms)} 个
    - 其他保护机制: {len(other_mechanisms)} 个
    
    ## 关键保护通路
    
    """
    
    # 添加关键保护基因信息
    if toxicity_result["mechanisms"]:
        report_text += "### 关键保护基因\n\n"
        
        # 排序保护机制，找出最重要的5个
        sorted_mechanisms = sorted(toxicity_result["mechanisms"], 
                                   key=lambda x: x["protection_score"], 
                                   reverse=True)
        
        for i, mechanism in enumerate(sorted_mechanisms[:5], 1):
            gene_name = mechanism["gene_name"]
            toxicity_name = mechanism["toxicity_name"]
            protection_score = mechanism["protection_score"]
            
            report_text += f"{i}. **{gene_name}**: 减轻{toxicity_name}，保护评分 {protection_score:.4f}\n"
            report_text += f"   - {args.protective_drug} 对该基因的作用: {'抑制' if mechanism['protective_drug_effect'] < 0 else '激活'}\n"
            report_text += f"   - {args.toxic_drug} 对该基因的作用: {'抑制' if mechanism['toxicity_drug_effect'] < 0 else '激活'}\n"
    else:
        report_text += "未发现明确的保护通路。\n"
    
    # 添加结论部分
    report_text += """
    ## 结论
    """
    
    if toxicity_result["score"] > 0.7:
        report_text += f"\n基于iKraph知识图谱分析，{args.protective_drug}很可能对{args.toxic_drug}的肝肾毒性具有显著的保护作用。"
    elif toxicity_result["score"] > 0.3:
        report_text += f"\n基于iKraph知识图谱分析，{args.protective_drug}可能对{args.toxic_drug}的肝肾毒性具有一定的保护作用。"
    else:
        report_text += f"\n基于iKraph知识图谱分析，{args.protective_drug}对{args.toxic_drug}的肝肾毒性保护作用不明显。"
    
    # 保存报告
    report_file = os.path.join(reports_dir, f"{base_filename}_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"肝肾毒性减轻分析报告已保存到 {report_file}")
    logger.info("肝肾毒性减轻分析流程完成")
    
    return True

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"开始处理时间: {start_time}")
    
    # 运行分析流程
    success = run_hepatorenal_toxicity_analysis(args)
    
    # 记录结束时间
    end_time = datetime.now()
    processing_time = end_time - start_time
    logger.info(f"结束处理时间: {end_time}")
    logger.info(f"总处理时间: {processing_time}")
    
    # 返回状态码
    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"分析过程中发生严重错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)