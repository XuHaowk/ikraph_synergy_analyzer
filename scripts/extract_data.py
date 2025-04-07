#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据提取脚本
从iKraph知识图谱中提取实体和关系数据
"""

import os
import sys
import argparse
import logging
import json  # Add this import - it was missing
import pandas as pd
import networkx as nx
import gc
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from config.settings import PATHS, DATA_LOADING
from core.data_loader import load_json_file, enable_large_dataset_processing
from core.entity_extractor import extract_keyword_entities, extract_entities_by_ids
from core.relation_extractor import extract_relations_with_entities, update_relation_entity_ids, add_entity_names_to_relations
from utils.file_utils import check_directories, save_entities_to_csv, save_relations_to_csv

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='iKraph Knowledge Graph Data Extraction Tool')
    
    # 目录参数
    parser.add_argument('--data_dir', type=str, default=PATHS['data_dir'], help='iKraph数据目录路径')
    parser.add_argument('--output_dir', type=str, default=PATHS['output_dir'], help='输出目录路径')
    
    # 提取参数
    parser.add_argument('--keywords', type=str, nargs='+', help='关键词列表')
    parser.add_argument('--drug_keywords', type=str, nargs='+', help='药物关键词列表')
    parser.add_argument('--disease_keywords', type=str, nargs='+', help='疾病关键词列表')
    parser.add_argument('--entity_types', type=str, nargs='+', default=[], help='实体类型列表')
    parser.add_argument('--relation_types', type=str, nargs='+', default=[], help='关系类型列表')
    parser.add_argument('--exact_match', action='store_true', help='使用精确匹配（默认为部分匹配）')
    
    # 性能参数
    parser.add_argument('--chunk_size', type=int, default=DATA_LOADING['chunk_size'], help='大文件处理的块大小')
    parser.add_argument('--buffer_size', type=int, default=DATA_LOADING['buffer_size'], help='文件读取缓冲区大小')
    parser.add_argument('--parallel_chunks', type=int, default=DATA_LOADING['parallel_chunks'], help='并行处理的块数')
    parser.add_argument('--process_count', type=int, default=DATA_LOADING['process_count'], help='并行处理的进程数')
    parser.add_argument('--low_memory', action='store_true', help='启用低内存模式')
    
    return parser.parse_args()

def run_extraction(args):
    """运行数据提取流程"""
    logger.info("启动数据提取流程...")
    
    # 启用大型数据集处理
    enable_large_dataset_processing()
    
    # 检查目录
    if not check_directories(args.data_dir, args.output_dir):
        logger.error("目录检查失败，退出")
        return False
    
    try:
        # 加载实体数据
        logger.info("加载实体数据...")
        nodes_file = os.path.join(args.data_dir, "NER_ID_dict_cap_final.json")
        nodes_data = load_json_file(
            nodes_file,
            chunk_size=args.chunk_size,
            low_memory=args.low_memory,
            method='auto'
        )
        
        if not nodes_data:
            logger.error("加载实体数据失败")
            return False
        
        # 提取实体
        logger.info("提取实体...")
        if args.drug_keywords or args.disease_keywords:
            # 提取药物实体
            drug_entities = []
            drug_id_map = {}
            
            if args.drug_keywords:
                # 处理多个药物关键词，逐个提取
                for drug_keyword in args.drug_keywords:
                    logger.info(f"提取药物: {drug_keyword}")
                    drug_entities_single, drug_id_map_single = extract_keyword_entities(
                        nodes_data,
                        keywords=[drug_keyword],  # 使用单个关键词
                        entity_types=["Chemical"],
                        exact_match=args.exact_match
                    )
                    drug_entities.extend(drug_entities_single)
                    drug_id_map.update(drug_id_map_single)
                
                logger.info(f"提取了 {len(drug_entities)} 个药物实体")
            
            # 提取疾病实体
            disease_entities = []
            disease_id_map = {}
            
            if args.disease_keywords:
                for disease_keyword in args.disease_keywords:
                    logger.info(f"提取疾病: {disease_keyword}")
                    disease_entities_single, disease_id_map_single = extract_keyword_entities(
                        nodes_data,
                        keywords=[disease_keyword],  # 使用单个关键词
                        entity_types=["Disease"],
                        exact_match=args.exact_match
                    )
                    disease_entities.extend(disease_entities_single)
                    disease_id_map.update(disease_id_map_single)
                
                logger.info(f"提取了 {len(disease_entities)} 个疾病实体")
            
            # 合并实体和ID映射
            all_entities = drug_entities + disease_entities
            entity_id_map = {**drug_id_map, **disease_id_map}
            
            # 创建焦点实体ID列表
            focal_ids = [entity["Original ID"] for entity in all_entities]
            
        elif args.keywords:
            # 使用通用关键词提取实体
            all_entities, entity_id_map = extract_keyword_entities(
                nodes_data,
                keywords=args.keywords,
                entity_types=args.entity_types if args.entity_types else None,
                exact_match=args.exact_match
            )
            logger.info(f"提取了 {len(all_entities)} 个实体")
            
            # 创建焦点实体ID列表
            focal_ids = [entity["Original ID"] for entity in all_entities]
        else:
            logger.error("未指定关键词，无法提取实体")
            return False
        
        # 保存焦点实体
        save_entities_to_csv(all_entities, args.output_dir, "focal_entities.csv")
        
        # 提取关系
        logger.info("提取关系...")
        
        # 加载关系类型模式
        schema_file = os.path.join(args.data_dir, "RelTypeInt.json")
        with open(schema_file, 'r', encoding='utf-8') as f:
            relation_schema = json.load(f)
        
        # 提取数据库关系
        logger.info("提取数据库关系...")
        db_file = os.path.join(args.data_dir, "DBRelations.json")
        db_data = load_json_file(
            db_file,
            chunk_size=args.chunk_size,
            low_memory=args.low_memory,
            method='auto'
        )
        
        db_relations = extract_relations_with_entities(
            db_data,
            focal_ids,
            relation_schema
        )
        logger.info(f"提取了 {len(db_relations)} 条数据库关系")
        
        # 释放内存
        del db_data
        gc.collect()
        
        # 提取PubMed关系
        logger.info("提取PubMed关系...")
        pubmed_file = os.path.join(args.data_dir, "PubMedList.json")
        
        # 由于PubMed文件通常很大，使用流式处理
        pubmed_relations = []
        for chunk in tqdm(load_json_file(pubmed_file, chunk_size=args.chunk_size, method='stream'), desc="处理PubMed关系"):
            chunk_relations = extract_relations_with_entities(
                chunk,
                focal_ids,
                relation_schema
            )
            pubmed_relations.extend(chunk_relations)
        
        logger.info(f"提取了 {len(pubmed_relations)} 条PubMed关系")
        
        # 合并关系
        all_relations = db_relations + pubmed_relations
        logger.info(f"共提取了 {len(all_relations)} 条关系")
        
        # 更新关系的实体ID
        updated_relations = update_relation_entity_ids(all_relations, entity_id_map)
        logger.info(f"更新了 {len(updated_relations)} 条关系的实体ID")
        
        # 添加实体名称
        relations_with_names = add_entity_names_to_relations(
            updated_relations,
            pd.DataFrame(all_entities)
        )
        
        # 保存关系
        save_relations_to_csv(relations_with_names, args.output_dir)
        
        # 提取关联实体
        logger.info("提取相关实体...")
        
        # 收集所有相关的实体ID
        related_ids = set()
        for relation in relations_with_names:
            related_ids.add(relation["Original Source ID"])
            related_ids.add(relation["Original Target ID"])
        
        # 移除焦点实体ID
        related_ids = related_ids - set(focal_ids)
        logger.info(f"发现 {len(related_ids)} 个相关实体")
        
        # 提取相关实体
        if related_ids:
            related_entities, related_id_map = extract_entities_by_ids(
                nodes_data,
                list(related_ids),
                args.entity_types if args.entity_types else None
            )
            
            # 保存相关实体
            save_entities_to_csv(related_entities, args.output_dir, "related_entities.csv")
            
            # 合并所有实体
            all_entities_combined = all_entities + related_entities
            
            # 保存所有实体
            save_entities_to_csv(all_entities_combined, args.output_dir, "all_entities.csv")
            
            logger.info(f"保存了 {len(all_entities_combined)} 个实体（包括 {len(all_entities)} 个焦点实体和 {len(related_entities)} 个相关实体）")
        
        logger.info("数据提取流程完成")
        return True
    
    except Exception as e:
        logger.error(f"数据提取过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False