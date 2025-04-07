#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
关系提取模块
提供从iKraph知识图谱中提取关系的功能
"""

import logging
import pandas as pd
from tqdm import tqdm
import gc
from typing import List, Dict, Tuple, Set, Union

logger = logging.getLogger(__name__)

def parse_relation_id(relation_id, relation_schema):
    """Parse relation ID with enhanced error handling and type normalization"""
    try:
        if not relation_id or not isinstance(relation_id, str):
            return None
            
        parts = relation_id.split(".")
        if len(parts) >= 6:
            source_id = parts[0]
            target_id = parts[1]
            relation_type_id = parts[2]
            correlation_id = parts[3]
            direction = parts[4]
            source = ".".join(parts[5:])  # Combine remaining parts as source
            
            # Get relation type name with fallback
            relation_type = "Unknown"
            
            # Try multiple ways to look up the relation type
            if relation_schema:
                # Try direct lookup
                if relation_type_id in relation_schema:
                    relation_type = relation_schema[relation_type_id]["name"]
                # Try as integer
                elif relation_type_id.isdigit() and int(relation_type_id) in relation_schema:
                    relation_type = relation_schema[int(relation_type_id)]["name"]
                # Try as string within a nested dictionary
                else:
                    for schema_id, schema_info in relation_schema.items():
                        if str(schema_id) == relation_type_id:
                            relation_type = schema_info["name"]
                            break
            
            return {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "relation_type_id": relation_type_id,
                "correlation_id": correlation_id,
                "direction": direction,
                "source": source
            }
        else:
            return None
    except Exception as e:
        return None

def extract_relations_with_entities(relations_data, entity_ids, relation_schema=None, min_confidence=0.0):
    """
    Extract relations connected to specific entities with multiple key strategies
    
    此函数增强了关系提取的灵活性和鲁棒性，支持多种数据结构和键名变体
    
    Parameters:
    - relations_data: 关系数据（列表或字典）
    - entity_ids: 目标实体ID列表
    - relation_schema: 关系模式（可选）
    - min_confidence: 最小置信度阈值
    
    Returns:
    - 提取的关系列表
    """
    logger.info(f"提取涉及特定实体的关系: 实体数量={len(entity_ids)}")
    
    # 特殊处理PubMed格式
    if isinstance(relations_data, list) and len(relations_data) > 0 and isinstance(relations_data[0], dict) and "id" in relations_data[0] and "list" in relations_data[0]:
        logger.info("检测到PubMed格式数据，使用专用提取逻辑")
        matched_relations = []
        
        # 创建实体ID集合用于快速查找
        entity_id_set = set(entity_ids)
        
        # 记录未映射的关系类型ID，用于调试
        unmapped_relation_types = set()
        
        for item in tqdm(relations_data, desc="处理PubMed关系"):
            rel_id = item.get("id")
            rel_data = item.get("list", [])
            
            # 跳过格式不正确的条目
            if not rel_id or not isinstance(rel_id, str) or not rel_data:
                continue
                
            # 解析关系ID
            parts = rel_id.split(".")
            if len(parts) < 2:
                continue
                
            source_id = parts[0]
            target_id = parts[1]
            
            # 检查此关系是否涉及我们感兴趣的实体
            if source_id not in entity_id_set and target_id not in entity_id_set:
                continue
                
            # 提取关系类型
            relation_type_id = parts[2] if len(parts) > 2 else "Unknown"
            relation_type = "Unknown"
            
            # 如果有关系模式，尝试解析关系类型
            if relation_schema and relation_type_id in relation_schema:
                if isinstance(relation_schema[relation_type_id], dict) and "name" in relation_schema[relation_type_id]:
                    relation_type = relation_schema[relation_type_id]["name"]
            
            if relation_type == "Unknown":
                unmapped_relation_types.add(relation_type_id)
            
            # 处理所有条目
            for entry in rel_data:
                if len(entry) >= 3:
                    # 提取得分和置信度，增强错误处理
                    try:
                        score = float(entry[0]) if entry[0] is not None and entry[0] != "" else 0.0
                        document = str(entry[1]) if len(entry) > 1 and entry[1] is not None else ""
                        confidence = float(entry[2]) if len(entry) > 2 and entry[2] is not None and entry[2] != "" else 0.0
                        
                        # 可选的新颖性字段
                        novelty = None
                        if len(entry) > 3 and entry[3] is not None:
                            try:
                                novelty = int(entry[3])
                            except (ValueError, TypeError):
                                pass
                    except (ValueError, TypeError):
                        # 跳过无效数据
                        continue
                    
                    # 检查置信度阈值
                    if min_confidence > 0 and confidence < min_confidence:
                        continue
                    
                    # 创建关系记录
                    relation = {
                        "Original Source ID": source_id,
                        "Original Target ID": target_id,
                        "Relation Type": relation_type,
                        "Relation Type ID": relation_type_id,
                        "Confidence": confidence,
                        "Score": score,
                        "Document": document,
                        "Source": "PubMed"
                    }
                    
                    if novelty is not None:
                        relation["Novelty"] = novelty
                        
                    matched_relations.append(relation)
        
        # 报告未映射的关系类型
        if unmapped_relation_types:
            logger.warning(f"发现 {len(unmapped_relation_types)} 个未映射的关系类型ID: {list(unmapped_relation_types)[:10]}")
            
        logger.info(f"从PubMed格式数据中提取了 {len(matched_relations)} 条关系")
        return matched_relations
    
    # 潜在的源ID和目标ID键名列表
    source_id_keys = [
        'Original Source ID', 'source_id', 'node_one_id', 
        'sourceID', 'source', 'node1_id'
    ]
    target_id_keys = [
        'Original Target ID', 'target_id', 'node_two_id', 
        'targetID', 'target', 'node2_id'
    ]
    
    # 创建实体ID集合用于快速查找
    entity_id_set = set(entity_ids)
    matched_relations = []
    
    # 记录未映射的关系类型ID，用于调试
    unmapped_relation_types = set()
    
    # 处理列表格式的关系数据
    if isinstance(relations_data, list):
        total_relations = len(relations_data)
        logger.info(f"处理列表格式的关系数据，总数: {total_relations}")
        
        for item in tqdm(relations_data, desc="处理关系列表", total=total_relations):
            if not isinstance(item, dict):
                continue
            
            # 灵活提取源ID和目标ID
            source_id = next((item.get(key) for key in source_id_keys if item.get(key)), None)
            target_id = next((item.get(key) for key in target_id_keys if item.get(key)), None)
            
            # 跳过缺少源或目标ID的关系
            if not source_id or not target_id:
                continue
            
            # 检查是否与目标实体相关
            if source_id not in entity_id_set and target_id not in entity_id_set:
                continue
            
            # 灵活提取关系类型
            relation_type_keys = ['relationship_type', 'Relation Type', 'type', 'relation_type']
            relation_type = next((item.get(key) for key in relation_type_keys if item.get(key)), "Unknown")
            
            # 灵活提取置信度和得分
            confidence = float(next((item.get(key) for key in ['prob', 'confidence', 'Confidence'] if item.get(key) is not None), 0.0))
            score = float(next((item.get(key) for key in ['score', 'Score'] if item.get(key) is not None), 0.0))
            
            # 应用置信度过滤
            if min_confidence > 0 and confidence < min_confidence:
                continue
            
            # 创建关系记录
            relation = {
                "Original Source ID": source_id,
                "Original Target ID": target_id,
                "Relation Type": relation_type,
                "Confidence": confidence,
                "Score": score,
                "Source": item.get("source", "Database")
            }
            
            # 尝试获取额外信息
            if "novelty" in item:
                relation["Novelty"] = item["novelty"]
            
            matched_relations.append(relation)
            
            # 记录未映射的关系类型
            if relation_type == "Unknown":
                unmapped_relation_types.add(str(item))
    
    # 处理字典格式的关系数据
    elif isinstance(relations_data, dict):
        total_relations = len(relations_data)
        logger.info(f"处理字典格式的关系数据，总数: {total_relations}")
        
        for rel_id, rel_data in tqdm(relations_data.items(), desc="处理关系字典", total=total_relations):
            if not isinstance(rel_data, dict):
                continue
            
            # 灵活提取源ID和目标ID
            source_id = next((rel_data.get(key) for key in source_id_keys if rel_data.get(key)), None)
            target_id = next((rel_data.get(key) for key in target_id_keys if rel_data.get(key)), None)
            
            # 尝试从关系ID解析源和目标ID
            if (not source_id or not target_id) and isinstance(rel_id, str):
                parts = rel_id.split('.')
                if len(parts) >= 2:
                    source_id = parts[0]
                    target_id = parts[1]
            
            # 跳过缺少源或目标ID的关系
            if not source_id or not target_id:
                continue
            
            # 检查是否与目标实体相关
            if source_id not in entity_id_set and target_id not in entity_id_set:
                continue
            
            # 灵活提取关系类型
            relation_type_keys = ['relationship_type', 'Relation Type', 'type', 'relation_type']
            relation_type = next((rel_data.get(key) for key in relation_type_keys if rel_data.get(key)), "Unknown")
            
            # 如果关系类型未找到，尝试从关系ID中提取
            if relation_type == "Unknown" and isinstance(rel_id, str):
                parts = rel_id.split('.')
                if len(parts) >= 3:
                    relation_type_id = parts[2]
                    # 如果有关系模式，尝试查找关系类型
                    if relation_schema and relation_type_id in relation_schema:
                        relation_type = relation_schema[relation_type_id]["name"]
            
            # 灵活提取置信度和得分
            confidence = float(next((rel_data.get(key) for key in ['prob', 'confidence', 'Confidence'] if rel_data.get(key) is not None), 0.0))
            score = float(next((rel_data.get(key) for key in ['score', 'Score'] if rel_data.get(key) is not None), 0.0))
            
            # 应用置信度过滤
            if min_confidence > 0 and confidence < min_confidence:
                continue
            
            # 创建关系记录
            relation = {
                "Original Source ID": source_id,
                "Original Target ID": target_id,
                "Relation Type": relation_type,
                "Confidence": confidence,
                "Score": score,
                "Source": rel_data.get("source", "Database")
            }
            
            # 尝试获取额外信息
            if "novelty" in rel_data:
                relation["Novelty"] = rel_data["novelty"]
            
            matched_relations.append(relation)
            
            # 记录未映射的关系类型
            if relation_type == "Unknown":
                unmapped_relation_types.add(str(rel_data))
    
    else:
        logger.error(f"不支持的关系数据类型: {type(relations_data)}")
        return []
    
    # 报告未映射的关系类型
    if unmapped_relation_types:
        logger.warning(f"发现 {len(unmapped_relation_types)} 个未映射的关系类型示例: {list(unmapped_relation_types)[:10]}")
    
    logger.info(f"提取了 {len(matched_relations)} 条与目标实体相关的关系")
    return matched_relations

def extract_single_pubmed_relation_rapidjson(relation_item, relation_schema):
    """Extract a single PubMed relation with enhanced error handling and relation type mapping"""
    results = []
    
    try:
        rel_id = relation_item.get("id")
        rel_data = relation_item.get("list", [])
        
        # Validate relation ID
        if not rel_id or not isinstance(rel_id, str):
            return results
            
        # Parse relation ID
        rel_components = parse_relation_id(rel_id, relation_schema)
        if not rel_components:
            return results
        
        # Get source and target IDs
        source_id = rel_components["source_id"]
        target_id = rel_components["target_id"]
        
        # Get relation type information
        relation_type_id = rel_components["relation_type_id"]
        relation_type = rel_components["relation_type"]
        
        # Process relation entries
        for entry in rel_data:
            if len(entry) >= 3:
                try:
                    # Extract data fields with better error handling
                    score = float(entry[0]) if entry[0] and isinstance(entry[0], (int, float, str)) else 0.0
                    document_id = str(entry[1]) if len(entry) > 1 and entry[1] else ""
                    confidence = float(entry[2]) if len(entry) > 2 and entry[2] and isinstance(entry[2], (int, float, str)) else 0.0
                    
                    # Handle novelty with error checking
                    novelty = None
                    if len(entry) > 3 and entry[3] is not None:
                        try:
                            novelty = int(entry[3])
                        except (ValueError, TypeError):
                            pass
                    
                    # Create relation record
                    relation = {
                        "Original Source ID": source_id,
                        "Original Target ID": target_id,
                        "Relation Type": relation_type,
                        "Relation Type ID": relation_type_id,
                        "Confidence": confidence,
                        "Score": score,
                        "Document": document_id,
                        "Novelty": novelty,
                        "Source": "PubMed",
                        "Direction": rel_components.get("direction", "forward")
                    }
                    
                    results.append(relation)
                    
                except (ValueError, TypeError, IndexError):
                    # Skip malformed entries
                    continue
    
    except Exception as e:
        # Log the error and return empty results
        logger.error(f"Error extracting relation: {e}")
        return []
    
    return results

def extract_pubmed_relations(pubmed_data, entity_id_map, relation_schema, rel_type_map, min_confidence=0.0):
    """提取PubMed关系数据，使用关系类型映射"""
    logger.info("从PubMed数据中提取关系...")
    relations = []
    
    # 记录未映射的关系类型ID，用于调试
    unmapped_ids = set()
    
    # 处理基于数据类型
    if isinstance(pubmed_data, list):
        logger.info(f"PubMed数据是一个列表结构，包含 {len(pubmed_data)} 个元素")
        
        # 遍历所有PubMed关系
        for item in tqdm(pubmed_data, desc="处理PubMed关系列表"):
            # 提取关系ID和数据
            if isinstance(item, dict) and "id" in item and "list" in item:
                rel_id = item["id"]
                rel_data = item["list"]
                
                # 解析关系ID
                rel_components = parse_relation_id(rel_id, relation_schema)
                if not rel_components:
                    continue
                
                # 检查实体是否在我们的映射中
                source_id = rel_components["source_id"]
                target_id = rel_components["target_id"]
                
                if source_id not in entity_id_map or target_id not in entity_id_map:
                    continue
                
                # 提取关系类型和关系类型ID
                relation_type_id = rel_components["relation_type_id"]
                
                # 尝试多种方式查找关系类型名称
                relation_type = None
                
                # 1. 直接从rel_type_map查找字符串ID
                if relation_type_id in rel_type_map:
                    relation_type = rel_type_map[relation_type_id]
                # 2. 尝试将ID转换为整数再查找
                elif relation_type_id.isdigit() and int(relation_type_id) in rel_type_map:
                    relation_type = rel_type_map[int(relation_type_id)]
                # 3. 如果上述方法都失败，使用解析出的关系类型
                else:
                    relation_type = rel_components["relation_type"]
                    # 记录未映射到的ID
                    unmapped_ids.add(relation_type_id)
                
                # 处理所有此关系的条目
                for entry in rel_data:
                    if len(entry) >= 3:  # 确保有足够的元素
                        # 创建关系记录
                        relation = {
                            "Source ID": entity_id_map[source_id],
                            "Target ID": entity_id_map[target_id],
                            "Relation Type": relation_type,
                            "Relation Type ID": relation_type_id,  # 保留ID以便参考
                            "Confidence": float(entry[2]) if len(entry) > 2 and entry[2] else 0.0,
                            "Score": float(entry[0]) if entry[0] else 0.0,
                            "Document": entry[1] if len(entry) > 1 else "",
                            "Novelty": int(entry[3]) if len(entry) > 3 and entry[3] is not None else None,
                            "Source": "PubMed",
                            "Original Source ID": source_id,
                            "Original Target ID": target_id,
                            "Direction": rel_components["direction"]
                        }
                        
                        # 应用置信度过滤
                        if min_confidence > 0 and relation["Confidence"] < min_confidence:
                            continue
                            
                        relations.append(relation)
    
    elif isinstance(pubmed_data, dict):
        logger.info("PubMed数据是一个字典结构")
        # 遍历所有PubMed关系
        for rel_id, rel_data in tqdm(pubmed_data.items(), desc="处理PubMed关系"):
            # 解析关系ID
            rel_components = parse_relation_id(rel_id, relation_schema)
            if not rel_components:
                continue
            
            # 检查实体是否在我们的映射中
            source_id = rel_components["source_id"]
            target_id = rel_components["target_id"]
            
            if source_id not in entity_id_map or target_id not in entity_id_map:
                continue
            
            # 提取关系类型和关系类型ID
            relation_type_id = rel_components["relation_type_id"]
            
            # 尝试多种方式查找关系类型名称
            relation_type = None
            
            # 1. 直接从rel_type_map查找字符串ID
            if relation_type_id in rel_type_map:
                relation_type = rel_type_map[relation_type_id]
            # 2. 尝试将ID转换为整数再查找
            elif relation_type_id.isdigit() and int(relation_type_id) in rel_type_map:
                relation_type = rel_type_map[int(relation_type_id)]
            # 3. 如果上述方法都失败，使用解析出的关系类型
            else:
                relation_type = rel_components["relation_type"]
                # 记录未映射到的ID
                unmapped_ids.add(relation_type_id)
            
            # 处理所有此关系的条目
            for entry in rel_data.get("list", []):
                if len(entry) >= 3:  # 确保有足够的元素
                    score = float(entry[0]) if entry[0] else 0.0
                    document_id = entry[1] if len(entry) > 1 else ""
                    confidence = float(entry[2]) if len(entry) > 2 and entry[2] else 0.0
                    novelty = int(entry[3]) if len(entry) > 3 and entry[3] is not None else None
                    
                    # 仅当设置了min_confidence时才应用置信度过滤
                    if min_confidence > 0 and confidence < min_confidence:
                        continue
                    
                    # 创建关系记录
                    relation = {
                        "Source ID": entity_id_map[source_id],
                        "Target ID": entity_id_map[target_id],
                        "Relation Type": relation_type,
                        "Relation Type ID": relation_type_id,
                        "Confidence": confidence,
                        "Score": score,
                        "Document": document_id,
                        "Novelty": novelty,
                        "Source": "PubMed",
                        "Original Source ID": source_id,
                        "Original Target ID": target_id,
                        "Direction": rel_components["direction"]
                    }
                    
                    relations.append(relation)
    else:
        logger.error(f"不支持的PubMed数据类型: {type(pubmed_data)}")
    
    # 报告未映射的关系类型ID
    if unmapped_ids:
        logger.warning(f"发现 {len(unmapped_ids)} 个未映射的关系类型ID: {', '.join(list(unmapped_ids)[:10])}" + 
                      (f"... 等" if len(unmapped_ids) > 10 else ""))
    
    logger.info(f"从PubMed数据中提取了 {len(relations)} 条关系")
    return relations

def extract_db_relations(db_data, entity_id_map, relation_schema, rel_type_map, min_confidence=0.0):
    """从数据库关系中提取关系，使用关系类型映射"""
    logger.info("从数据库数据中提取关系...")
    relations = []
    
    # 记录未映射的关系类型ID，用于调试
    unmapped_ids = set()
    
    # 确认数据类型并适当处理
    if isinstance(db_data, list):
        logger.info(f"数据库关系是一个列表结构，包含 {len(db_data)} 个元素")
        
        # 遍历所有数据库关系
        for item in tqdm(db_data, desc="处理数据库关系列表"):
            if not isinstance(item, dict):
                continue
                
            # 获取源和目标ID
            source_id = item.get("node_one_id")
            target_id = item.get("node_two_id")
            
            # 检查实体是否在我们的映射中
            if not source_id or not target_id:
                continue
            
            if source_id not in entity_id_map or target_id not in entity_id_map:
                continue
            
            # 提取关系属性
            relation_type_raw = item.get("relationship_type", "Unknown")
            
            # 尝试多种方式查找关系类型名称
            relation_type = None
            
            # 检查关系类型是否为数字（ID）
            if relation_type_raw.isdigit():
                # 1. 直接从rel_type_map查找字符串ID
                if relation_type_raw in rel_type_map:
                    relation_type = rel_type_map[relation_type_raw]
                # 2. 尝试将ID转换为整数再查找
                elif int(relation_type_raw) in rel_type_map:
                    relation_type = rel_type_map[int(relation_type_raw)]
                else:
                    relation_type = "Unknown"
                    # 记录未映射到的ID
                    unmapped_ids.add(relation_type_raw)
            else:
                relation_type = relation_type_raw
            
            # 获取置信度和得分
            confidence = float(item.get("prob", 0.0)) if item.get("prob") else 0.0
            score = float(item.get("score", 0.0)) if item.get("score") else 0.0
            
            # 应用置信度过滤
            if min_confidence > 0 and confidence < min_confidence:
                continue
            
            # 创建关系记录
            relation = {
                "Source ID": entity_id_map[source_id],
                "Target ID": entity_id_map[target_id],
                "Relation Type": relation_type,
                "Relation Type ID": relation_type_raw if relation_type_raw.isdigit() else "",
                "Confidence": confidence,
                "Score": score,
                "Source": item.get("source", "Database"),
                "Original Source ID": source_id,
                "Original Target ID": target_id
            }
            
            relations.append(relation)
    
    elif isinstance(db_data, dict):
        logger.info("数据库关系是一个字典结构")
        # 遍历所有数据库关系
        for rel_id, rel_data in tqdm(db_data.items(), desc="处理数据库关系"):
            # 获取源和目标ID
            source_id = rel_data.get("node_one_id")
            target_id = rel_data.get("node_two_id")
            
            # 如果IDs没有在rel_data中，尝试从rel_id提取
            if (not source_id or not target_id) and isinstance(rel_id, str):
                parts = rel_id.split('.')
                if len(parts) >= 2:
                    source_id = parts[0]
                    target_id = parts[1]
            
            # 检查实体是否在我们的映射中
            if not source_id or not target_id:
                continue
            
            if source_id not in entity_id_map or target_id not in entity_id_map:
                continue
            
            # 提取关系属性
            relation_type_raw = rel_data.get("relationship_type", "Unknown")
            
            # 如果关系类型未找到，尝试从rel_id中提取
            if relation_type_raw == "Unknown" and isinstance(rel_id, str):
                parts = rel_id.split('.')
                if len(parts) >= 3:
                    relation_type_id = parts[2]
                    if relation_type_id.isdigit():
                        relation_type_raw = relation_type_id
            
            # 尝试多种方式查找关系类型名称
            relation_type = None
            
            # 检查关系类型是否为数字（ID）
            if isinstance(relation_type_raw, str) and relation_type_raw.isdigit():
                # 1. 直接从rel_type_map查找字符串ID
                if relation_type_raw in rel_type_map:
                    relation_type = rel_type_map[relation_type_raw]
                # 2. 尝试将ID转换为整数再查找
                elif int(relation_type_raw) in rel_type_map:
                    relation_type = rel_type_map[int(relation_type_raw)]
                else:
                    relation_type = "Unknown"
                    # 记录未映射到的ID
                    unmapped_ids.add(relation_type_raw)
            else:
                relation_type = relation_type_raw
            
            # 获取置信度和得分
            confidence = float(rel_data.get("prob", 0.0)) if rel_data.get("prob") else 0.0
            score = float(rel_data.get("score", 0.0)) if rel_data.get("score") else 0.0
            
            # 应用置信度过滤
            if min_confidence > 0 and confidence < min_confidence:
                continue
            
            # 创建关系记录
            relation = {
                "Source ID": entity_id_map[source_id],
                "Target ID": entity_id_map[target_id],
                "Relation Type": relation_type,
                "Relation Type ID": relation_type_raw if isinstance(relation_type_raw, str) and relation_type_raw.isdigit() else "",
                "Confidence": confidence,
                "Score": score,
                "Source": rel_data.get("source", "Database"),
                "Original Source ID": source_id,
                "Original Target ID": target_id
            }
            
            relations.append(relation)
    else:
        logger.error(f"不支持的数据库关系数据类型: {type(db_data)}")
    
    # 报告未映射的关系类型ID
    if unmapped_ids:
        logger.warning(f"发现 {len(unmapped_ids)} 个未映射的关系类型ID: {', '.join(list(unmapped_ids)[:10])}" + 
                      (f"... 等" if len(unmapped_ids) > 10 else ""))
    
    logger.info(f"从数据库关系中提取了 {len(relations)} 个关系")
    return relations

def add_entity_names_to_relations(relations, entities_df):
    """向关系添加实体名称，优先使用通用名称，清理中文字符但保留希腊字母"""
    logger.info("向关系添加实体名称...")
    
    # 创建ID到名称的映射
    id_to_official_name = dict(zip(entities_df["ID"], entities_df["Official_Name"]))
    id_to_common_name = dict(zip(entities_df["ID"], entities_df["Common_Name"]))
    id_to_type = dict(zip(entities_df["ID"], entities_df["Type"]))
    
    # 添加名称和类型到关系
    for relation in tqdm(relations, desc="向关系添加实体名称"):
        source_id = relation["Source ID"]
        target_id = relation["Target ID"]
        
        # 优先使用通用名称，如果没有则使用官方名称
        source_common = id_to_common_name.get(source_id, "")
        source_official = id_to_official_name.get(source_id, "")
        target_common = id_to_common_name.get(target_id, "")
        target_official = id_to_official_name.get(target_id, "")
        
        # 使用清理函数处理名称，保留希腊字母
        relation["Source Name"] = source_common if source_common else source_official
        relation["Source Official Name"] = source_official
        relation["Target Name"] = target_common if target_common else target_official
        relation["Target Official Name"] = target_official
        relation["Source Type"] = id_to_type.get(source_id, "Unknown")
        relation["Target Type"] = id_to_type.get(target_id, "Unknown")
    
    return relations

def update_relation_entity_ids(relations, entity_id_map):
    """更新关系的实体ID映射"""
    updated_relations = []
    
    for relation in tqdm(relations, desc="更新关系实体ID"):
        # 获取原始ID
        source_id = relation.get("Original Source ID")
        target_id = relation.get("Original Target ID")
        
        # 检查ID是否在映射中
        if source_id in entity_id_map and target_id in entity_id_map:
            # 更新ID
            updated_relation = relation.copy()
            updated_relation["Source ID"] = entity_id_map[source_id]
            updated_relation["Target ID"] = entity_id_map[target_id]
            updated_relations.append(updated_relation)
    
    logger.info(f"更新了 {len(updated_relations)} 条关系的实体ID")
    return updated_relations

def merge_duplicate_relations(relations):
    """
    根据iKraph结构合并重复的关系记录，忽略大小写差异
    """
    logger.info("正在合并重复的关系记录...")
    
    # 记录合并前的关系数量
    original_count = len(relations)
    
    # 用于标识唯一关系的键（标准化源ID、目标ID和关系类型，包括大小写处理）
    grouped_relations = {}
    
    # 按源实体、目标实体和关系类型分组
    for relation in tqdm(relations, desc="合并重复关系"):
        # 转换关系类型为小写以忽略大小写差异
        rel_type = relation.get("Relation Type", "")
        rel_type_lower = rel_type.lower() if isinstance(rel_type, str) else rel_type
        
        key = (relation.get("Original Source ID", ""), 
               relation.get("Original Target ID", ""), 
               rel_type_lower)  # 关系类型转为小写
        
        # 跳过缺少关键数据的关系
        if not all(key):
            continue
            
        if key not in grouped_relations:
            # 第一次遇到此关系，创建新记录
            base_record = {k: v for k, v in relation.items() 
                          if k not in ["Document", "Score", "Confidence", "Novelty"]}
            
            # 保留原始大小写格式的关系类型
            base_record["Original_Relation_Type"] = rel_type
            base_record["Evidence"] = []
            base_record["Max_Confidence"] = 0
            base_record["Avg_Confidence"] = 0
            base_record["Evidence_Count"] = 0
            grouped_relations[key] = base_record
        else:
            # 如果当前关系类型有更好的格式（比如首字母大写而不是全大写或全小写）
            current_rel_type = relation.get("Relation Type", "")
            if isinstance(current_rel_type, str):
                curr_lower = current_rel_type.lower()
                curr_upper = current_rel_type.upper()
                
                # 检查当前值是否比保存值更合适（首字母大写优于全大写或全小写）
                if current_rel_type != curr_lower and current_rel_type != curr_upper:
                    grouped_relations[key]["Original_Relation_Type"] = current_rel_type
        
        # 添加证据
        evidence = {
            "Document": relation.get("Document", ""),
            "Score": relation.get("Score", 0),
            "Confidence": relation.get("Confidence", 0),
            "Novelty": relation.get("Novelty", None)
        }
        
        # 更新聚合统计
        conf = float(evidence["Confidence"]) if evidence["Confidence"] else 0
        grouped_relations[key]["Evidence"].append(evidence)
        grouped_relations[key]["Evidence_Count"] += 1
        grouped_relations[key]["Max_Confidence"] = max(grouped_relations[key]["Max_Confidence"], conf)
        
        # 更新平均置信度
        total = grouped_relations[key]["Avg_Confidence"] * (grouped_relations[key]["Evidence_Count"] - 1)
        grouped_relations[key]["Avg_Confidence"] = (total + conf) / grouped_relations[key]["Evidence_Count"]
    
    # 转换回列表并格式化输出
    result = []
    for key, rel_data in grouped_relations.items():
        # 选择最高置信度的证据作为主要显示
        best_evidence = max(rel_data["Evidence"], 
                           key=lambda e: e["Confidence"] if e["Confidence"] else 0) if rel_data["Evidence"] else {}
        
        # 创建输出记录
        output_record = {k: v for k, v in rel_data.items() 
                        if k not in ["Evidence", "Avg_Confidence"]}
        
        # 使用更好格式的关系类型
        if "Original_Relation_Type" in output_record:
            output_record["Relation Type"] = output_record["Original_Relation_Type"]
            del output_record["Original_Relation_Type"]
        
        # 添加主要证据信息
        output_record["Confidence"] = rel_data["Max_Confidence"]
        output_record["Score"] = best_evidence.get("Score", 0)
        output_record["Document"] = best_evidence.get("Document", "")
        if "Novelty" in best_evidence and best_evidence["Novelty"] is not None:
            output_record["Novelty"] = best_evidence["Novelty"]
        
        # 添加证据摘要信息
        output_record["Evidence_Summary"] = f"{rel_data['Evidence_Count']} references"
        if rel_data["Evidence_Count"] > 1:
            output_record["Other_Documents"] = "|".join(str(e.get("Document", "")) 
                                                      for e in rel_data["Evidence"][:5] 
                                                      if e.get("Document") != output_record["Document"])
            if len(rel_data["Evidence"]) > 5:
                output_record["Other_Documents"] += f"... (+{len(rel_data['Evidence'])-5} more)"
        
        result.append(output_record)
    
    logger.info(f"合并完成：从 {original_count} 条关系记录合并为 {len(result)} 条")
    return result

def extract_relations_with_direction(relations_data, entity_ids, relation_schema, rel_type_map=None, min_confidence=0.0):
    """
    提取带有方向信息的关系
    
    Parameters:
    - relations_data: 关系数据
    - entity_ids: 实体ID列表
    - relation_schema: 关系模式，用于确定关系方向
    - rel_type_map: 关系类型映射
    - min_confidence: 最小置信度
    
    Returns:
    - 带有方向信息的关系列表
    """
    # 首先提取关系
    relations = extract_relations_with_entities(relations_data, entity_ids, relation_schema, min_confidence)
    
    # 添加方向信息
    for relation in relations:
        relation_type_id = relation.get("Relation Type ID", "")
        if not relation_type_id and "Relation Type" in relation:
            # 尝试根据关系类型名称反向查找ID
            if rel_type_map:
                for rid, rname in rel_type_map.items():
                    if rname == relation["Relation Type"]:
                        relation_type_id = rid
                        relation["Relation Type ID"] = rid
                        break
        
        # 设置方向
        if relation_type_id and relation_type_id in relation_schema:
            relation["Direction"] = relation_schema[relation_type_id].get("direction", 0)
        elif "Relation Type" in relation:
            # 根据关系类型名称确定方向
            rel_type = relation["Relation Type"].lower()
            if "positive" in rel_type or "activates" in rel_type or "increases" in rel_type:
                relation["Direction"] = 1
            elif "negative" in rel_type or "inhibits" in rel_type or "decreases" in rel_type:
                relation["Direction"] = -1
            else:
                relation["Direction"] = 0
    
    return relations

def filter_relations_by_type(relations, relation_types):
    """
    按关系类型过滤关系
    
    Parameters:
    - relations: 关系列表
    - relation_types: 关系类型列表
    
    Returns:
    - 过滤后的关系列表
    """
    if not relation_types:
        return relations
    
    filtered = []
    relation_type_set = set(relation_types)
    
    for relation in relations:
        rel_type = relation.get("Relation Type")
        rel_type_id = relation.get("Relation Type ID")
        
        if rel_type in relation_type_set or rel_type_id in relation_type_set:
            filtered.append(relation)
    
    logger.info(f"按关系类型过滤: 从 {len(relations)} 条关系中筛选出 {len(filtered)} 条")
    return filtered

def filter_relations_by_entity_types(relations, entity_type_pairs):
    """
    按实体类型对过滤关系
    
    Parameters:
    - relations: 关系列表
    - entity_type_pairs: 实体类型对列表，如[("Drug", "Gene"), ("Gene", "Disease")]
    
    Returns:
    - 过滤后的关系列表
    """
    if not entity_type_pairs:
        return relations
    
    filtered = []
    
    for relation in relations:
        source_type = relation.get("Source Type")
        target_type = relation.get("Target Type")
        
        if (source_type, target_type) in entity_type_pairs or (target_type, source_type) in entity_type_pairs:
            filtered.append(relation)
    
    logger.info(f"按实体类型对过滤: 从 {len(relations)} 条关系中筛选出 {len(filtered)} 条")
    return filtered 


