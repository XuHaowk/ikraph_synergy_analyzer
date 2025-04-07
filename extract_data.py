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
import json
import pandas as pd
import networkx as nx
import gc
from tqdm import tqdm
from datetime import datetime

# 调试输出
print("数据提取脚本启动...")

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
    print("原始命令行参数:", sys.argv)
    
    parser = argparse.ArgumentParser(description='iKGraph Knowledge Graph Data Extraction Tool')
    
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
    
    args = parser.parse_args()
    
    # 添加调试输出
    print("解析后的参数:")
    print("数据目录:", args.data_dir)
    print("输出目录:", args.output_dir)
    print("药物关键词:", args.drug_keywords)
    print("疾病关键词:", args.disease_keywords)
    print("精确匹配:", args.exact_match)
    print("检查数据目录:")
    print("目录存在:", os.path.exists(args.data_dir))
    if os.path.exists(args.data_dir):
        print("目录内容:", os.listdir(args.data_dir))
    
    return args

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
                    print(f"正在提取药物: {drug_keyword}")
                    drug_entities_single, drug_id_map_single = extract_keyword_entities(
                        nodes_data,
                        keywords=[drug_keyword],  # 使用单个关键词
                        entity_types=["Chemical"],
                        exact_match=args.exact_match
                    )
                    drug_entities.extend(drug_entities_single)
                    drug_id_map.update(drug_id_map_single)
                
                logger.info(f"提取了 {len(drug_entities)} 个药物实体")
                print(f"提取了 {len(drug_entities)} 个药物实体")
            
            # 提取疾病实体
            disease_entities = []
            disease_id_map = {}
            
            if args.disease_keywords:
                for disease_keyword in args.disease_keywords:
                    logger.info(f"提取疾病: {disease_keyword}")
                    print(f"正在提取疾病: {disease_keyword}")
                    disease_entities_single, disease_id_map_single = extract_keyword_entities(
                        nodes_data,
                        keywords=[disease_keyword],  # 使用单个关键词
                        entity_types=["Disease"],
                        exact_match=args.exact_match
                    )
                    disease_entities.extend(disease_entities_single)
                    disease_id_map.update(disease_id_map_single)
                
                logger.info(f"提取了 {len(disease_entities)} 个疾病实体")
                print(f"提取了 {len(disease_entities)} 个疾病实体")
            
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
        logger.info(f"已保存 {len(all_entities)} 个焦点实体")
        print(f"已保存 {len(all_entities)} 个焦点实体")
        
        # 提取关系
        logger.info("提取关系...")
        print("开始提取关系...")
        
        # 加载关系类型模式
        schema_file = os.path.join(args.data_dir, "RelTypeInt.json")
        print(f"尝试加载关系模式文件: {schema_file}")
        print(f"文件存在: {os.path.exists(schema_file)}")
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                relation_schema = json.load(f)
            print(f"成功加载关系模式，包含 {len(relation_schema)} 种关系类型")
        except Exception as e:
            print(f"加载关系模式失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
        
        # 提取数据库关系
        logger.info("提取数据库关系...")
        db_file = os.path.join(args.data_dir, "DBRelations.json")
        print(f"尝试加载数据库关系文件: {db_file}")
        print(f"文件存在: {os.path.exists(db_file)}")
        
        db_data = load_json_file(
            db_file,
            chunk_size=args.chunk_size,
            low_memory=args.low_memory,
            method='auto'
        )
        
        print(f"加载的数据库关系数量: {len(db_data) if db_data else 0}")
        
        if db_data is None:
            print("数据库关系加载失败，跳过此步骤")
            db_relations = []
        else:
            db_relations = extract_relations_with_entities(
                db_data,
                focal_ids,
                relation_schema
            )
            logger.info(f"提取了 {len(db_relations)} 条数据库关系")
            print(f"提取了 {len(db_relations)} 条数据库关系")
        
        # 释放内存
        del db_data
        gc.collect()
        
        # 提取PubMed关系
        logger.info("提取PubMed关系...")
        pubmed_file = os.path.join(args.data_dir, "PubMedList.json")
        print(f"尝试加载PubMed关系文件: {pubmed_file}")
        print(f"文件存在: {os.path.exists(pubmed_file)}")

        try:
            # 跳过整个PubMed流式处理，转而使用batch_main_xin_4_7.py中的并行处理方法
            from core.data_loader import parallel_process_pubmed_relations
    
            # 创建一个更简单的focal_entities结构
            simple_focal_entities = {
                'drug': [entity["Original ID"] for entity in drug_entities],
                'disease': [entity["Original ID"] for entity in disease_entities]
            }
    
            num_processes = min(12, os.cpu_count())  # 限制进程数以避免内存问题
            pubmed_relations, _ = parallel_process_pubmed_relations(
                pubmed_file,
                focal_ids,
                simple_focal_entities,
                relation_schema,
                num_processes=num_processes,
                chunk_size=args.chunk_size,
                buffer_size=50*1024*1024  # 50MB缓冲区
            )
    
            logger.info(f"提取了 {len(pubmed_relations)} 条PubMed关系")
            print(f"提取了 {len(pubmed_relations)} 条PubMed关系")
        except Exception as e:
            print(f"处理PubMed关系时出错: {e}")
            import traceback
            print(traceback.format_exc())
            pubmed_relations = []
        
        # 合并关系
        all_relations = db_relations + pubmed_relations
        logger.info(f"共提取了 {len(all_relations)} 条关系")
        print(f"共提取了 {len(all_relations)} 条关系")

        # 直接处理关系ID并保存
        if all_relations:
            # 创建一个简单的自动ID映射函数
            print("使用自动ID映射...")
            # 为源和目标实体分配新的ID
            entity_mapping = {}
            next_id = 1
    
            # 首先为焦点实体分配ID
            for focal_id in focal_ids:
                entity_mapping[focal_id] = next_id
                next_id += 1
    
            # 再为关系中涉及的其他实体分配ID
            for relation in all_relations:
                source_id = relation.get("Original Source ID")
                target_id = relation.get("Original Target ID")
        
                if source_id and source_id not in entity_mapping:
                    entity_mapping[source_id] = next_id
                    next_id += 1
            
                if target_id and target_id not in entity_mapping:
                    entity_mapping[target_id] = next_id
                    next_id += 1
    
            # 更新关系的实体ID
            updated_relations = []
            for relation in all_relations:
                source_id = relation.get("Original Source ID")
                target_id = relation.get("Original Target ID")
        
                if source_id in entity_mapping and target_id in entity_mapping:
                    relation_copy = relation.copy()
                    relation_copy["Source ID"] = entity_mapping[source_id]
                    relation_copy["Target ID"] = entity_mapping[target_id]
                    updated_relations.append(relation_copy)
    
            print(f"自动映射了 {len(entity_mapping)} 个实体ID")
            print(f"更新了 {len(updated_relations)} 条关系的实体ID")
    
            # 保存关系
            save_relations_to_csv(updated_relations, args.output_dir)
            print(f"保存了 {len(updated_relations)} 条关系")
        
        # 提取关联实体
        logger.info("提取相关实体...")
        print("开始提取相关实体...")
        
        # 收集所有相关的实体ID
        # 收集所有相关的实体ID
        related_ids = set()
        for relation in updated_relations:  # Use updated_relations instead
            related_ids.add(relation["Original Source ID"])
            related_ids.add(relation["Original Target ID"])
        
        # 移除焦点实体ID
        related_ids = related_ids - set(focal_ids)
        logger.info(f"发现 {len(related_ids)} 个相关实体")
        print(f"发现 {len(related_ids)} 个相关实体")
        
        # 提取相关实体
        if related_ids:
            related_entities, related_id_map = extract_entities_by_ids(
                nodes_data,
                list(related_ids),
                args.entity_types if args.entity_types else None
            )
            
            # 保存相关实体
            save_entities_to_csv(related_entities, args.output_dir, "related_entities.csv")
            print(f"已保存 {len(related_entities)} 个相关实体")
            
            # 合并所有实体
            all_entities_combined = all_entities + related_entities
            
            # 保存所有实体
            save_entities_to_csv(all_entities_combined, args.output_dir, "all_entities.csv")
            
            logger.info(f"保存了 {len(all_entities_combined)} 个实体（包括 {len(all_entities)} 个焦点实体和 {len(related_entities)} 个相关实体）")
            print(f"保存了总共 {len(all_entities_combined)} 个实体")
        else:
            # 如果没有相关实体，直接使用焦点实体作为所有实体
            save_entities_to_csv(all_entities, args.output_dir, "all_entities.csv")
            print("没有找到相关实体，使用焦点实体作为所有实体")
        
        logger.info("数据提取流程完成")
        return True
    
    except Exception as e:
        print(f"严重错误: {e}")
        import traceback
        print(traceback.format_exc())
        logger.error(f"数据提取过程中发生错误: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"开始处理时间: {start_time}")
    
    # 运行提取流程
    success = run_extraction(args)
    
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
        print(f"CRITICAL ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)




