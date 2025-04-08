#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实体提取模块
提供从iKraph知识图谱中提取实体的功能
"""

import logging
import pandas as pd
from tqdm import tqdm
import gc
from typing import List, Dict, Tuple, Set, Union

logger = logging.getLogger(__name__)

def clean_non_ascii_chars(text, preserve_greek=True):
    """
    高级文本清理，使用Unicode规范化处理希腊字母
    """
    if not isinstance(text, str):
        return text
    
    import unicodedata
    import re
    
    # 选项1：使用Unicode规范化（保留希腊字母实际形式）
    if preserve_greek:
        # 基于Unicode块确定希腊字母
        def is_greek(char):
            return 'GREEK' in unicodedata.name(char, '')
        
        # 处理文本，保留希腊字母
        result = ""
        for char in text:
            if char.isascii() or is_greek(char):
                result += char
            else:
                # 尝试用NFKD分解获取ASCII等价形式
                normalized = unicodedata.normalize('NFKD', char)
                if all(c.isascii() for c in normalized):
                    result += normalized
        
        # 清理多余空格
        return re.sub(r'\s+', ' ', result).strip()
    else:
        # 如果不需要保留希腊字母，使用简单的ASCII过滤
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
        return re.sub(r'\s+', ' ', cleaned_text).strip()

def extract_entities_from_nodes(nodes_data, keywords=None, entity_types=None, exact_match=False, low_memory=False):
    """Extract entities from node data, with option for exact matching and cleaning Chinese characters"""
    logger.info("Extracting entities from node data...")
    
    # If keywords provided, convert to lowercase for case-insensitive matching
    if keywords:
        keywords = [k.lower() for k in keywords]
    
    # Ensure nodes_data is a list
    if not isinstance(nodes_data, list):
        logger.error(f"Unexpected data type: {type(nodes_data)}")
        return [], {}
    
    entities = []
    entity_id_map = {}  # For mapping biokdeid to new sequential ID
    
    # Process all entities
    total_entities = len(nodes_data)
    logger.info(f"Total entity count: {total_entities}")
    
    # Process keyword entities first
    keyword_entities = []
    non_keyword_entities = []
    
    for node in tqdm(nodes_data, desc="Processing entities", total=total_entities):
        # Ensure node is a dictionary
        if not isinstance(node, dict):
            logger.warning(f"Skipping non-dictionary node: {type(node)}")
            continue
        
        # Check if it's a keyword entity using exact or partial matching
        is_keyword = False
        if keywords:
            official_name = str(node.get("official name", "")).lower()
            common_name = str(node.get("common name", "")).lower()
            
            for keyword in keywords:
                if exact_match:
                    # Exact match (case-insensitive)
                    if keyword == official_name or keyword == common_name:
                        is_keyword = True
                        break
                else:
                    # Partial match (case-insensitive)
                    if keyword in official_name or keyword in common_name:
                        is_keyword = True
                        break
        
        # Filter entity types
        entity_type = node.get("type", "")
        if entity_types and entity_type not in entity_types and not is_keyword:
            continue
        
        # Clean names of Chinese characters, but preserve Greek letters
        cleaned_official_name = clean_non_ascii_chars(node.get("official name", ""), preserve_greek=True)
        cleaned_common_name = clean_non_ascii_chars(node.get("common name", ""), preserve_greek=True)
        
        # Create entity record
        entity = {
            "Original ID": node.get("biokdeid", ""),
            "Name": cleaned_common_name if cleaned_common_name else cleaned_official_name,
            "Official_Name": cleaned_official_name,
            "Common_Name": cleaned_common_name,
            "Type": entity_type,
            "Subtype": node.get("subtype", ""),
            "External ID": node.get("id", ""),
            "Species": ", ".join(map(str, node.get("species", []))) if node.get("species") else "",
            "Is Keyword": is_keyword
        }
        
        if is_keyword:
            keyword_entities.append(entity)
        elif entity_types and entity_type in entity_types:
            non_keyword_entities.append(entity)
        
        # In low memory mode, periodically clean up
        if low_memory and len(non_keyword_entities) % 1000000 == 0:
            gc.collect()
    
    # Merge keyword entities and other entities
    logger.info(f"Found {len(keyword_entities)} keyword entities and {len(non_keyword_entities)} non-keyword entities")
    all_entities = keyword_entities + non_keyword_entities
    
    # Assign sequential IDs to entities
    for i, entity in enumerate(all_entities):
        entity["ID"] = i + 1
        entity_id_map[entity["Original ID"]] = entity["ID"]
    
    logger.info(f"Extracted {len(all_entities)} entities (including {len(keyword_entities)} keyword entities)")
    return all_entities, entity_id_map

def extract_keyword_entities(nodes_data, keywords=None, entity_types=None, exact_match=False):
    """
    从节点数据中提取符合特定关键词和实体类型的实体，增强对中文关键词的处理
    
    Parameters:
    - nodes_data: 节点数据列表
    - keywords: 关键词列表
    - entity_types: 实体类型列表
    - exact_match: 是否使用精确匹配（默认为False，使用部分匹配）
    
    Returns:
    - 符合条件的实体列表和ID映射
    """
    logger.info(f"提取关键词实体: 关键词={keywords}, 实体类型={entity_types}, 精确匹配={exact_match}")
    
    # 如果提供了关键词，为中文保留原始大小写
    if keywords:
        # 对于中文关键词，我们不转换为小写，因为它不适用
        keywords_for_matching = []
        for k in keywords:
            # 检查关键词是否包含中文字符（Unicode范围检查）
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in k)
            if has_chinese:
                keywords_for_matching.append(k)  # 保留中文原始形式
            else:
                keywords_for_matching.append(k.lower())  # 非中文转换为小写
        
        keywords = keywords_for_matching  # 将处理后的关键词列表赋值回keywords变量
    
    # 确保nodes_data是列表
    if not isinstance(nodes_data, list):
        logger.error(f"意外的数据类型: {type(nodes_data)}")
        return [], {}
    
    matched_entities = []
    entity_id_map = {}  # 用于映射biokdeid到新的顺序ID
    
    # 遍历所有节点
    total_nodes = len(nodes_data)
    for node in tqdm(nodes_data, desc="处理关键词实体", total=total_nodes):
        # 确保节点是字典
        if not isinstance(node, dict):
            continue
        
        # 获取实体名称
        original_official_name = str(node.get("official name", ""))
        original_common_name = str(node.get("common name", ""))
        
        # 获取实体类型
        entity_type = node.get("type", "")
        
        # 检查实体类型是否匹配
        type_match = not entity_types or entity_type in entity_types
        if not type_match:
            continue
        
        # 检查关键词是否匹配
        keyword_match = False
        if keywords:
            for keyword in keywords:
                # 检查关键词是否含有中文
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in keyword)
                
                if exact_match:
                    # 对于中文字符，我们进行直接比较
                    if has_chinese:
                        if keyword == original_official_name or keyword == original_common_name:
                            keyword_match = True
                            break
                    else:
                        # 对于非中文，不区分大小写匹配
                        official_name_lower = original_official_name.lower()
                        common_name_lower = original_common_name.lower()
                        if keyword.lower() == official_name_lower or keyword.lower() == common_name_lower:
                            keyword_match = True
                            break
                else:
                    # 部分匹配
                    if has_chinese:
                        # 中文部分匹配需要原始形式
                        if keyword in original_official_name or keyword in original_common_name:
                            keyword_match = True
                            break
                    else:
                        # 非中文部分匹配使用小写形式
                        official_name_lower = original_official_name.lower()
                        common_name_lower = original_common_name.lower()
                        if keyword.lower() in official_name_lower or keyword.lower() in common_name_lower:
                            keyword_match = True
                            break
        else:
            # 如果没有提供关键词，则默认匹配所有实体类型符合的实体
            keyword_match = True
        
        # 如果同时满足类型和关键词匹配条件
        if type_match and keyword_match:
            # 清理名称中的非ASCII字符，但保留希腊字母
            cleaned_official_name = clean_non_ascii_chars(original_official_name, preserve_greek=True)
            cleaned_common_name = clean_non_ascii_chars(original_common_name, preserve_greek=True)
            
            # 创建实体记录
            entity = {
                "Original ID": node.get("biokdeid", ""),
                "Name": cleaned_common_name if cleaned_common_name else cleaned_official_name,
                "Official_Name": cleaned_official_name,
                "Common_Name": cleaned_common_name,
                "Type": entity_type,
                "Subtype": node.get("subtype", ""),
                "External ID": node.get("id", ""),
                "Species": ", ".join(map(str, node.get("species", []))) if node.get("species") else "",
                "Is Keyword": True  # 标记为关键词实体
            }
            
            matched_entities.append(entity)
    
    # 为匹配的实体分配顺序ID
    for i, entity in enumerate(matched_entities):
        entity["ID"] = i + 1
        entity_id_map[entity["Original ID"]] = entity["ID"]
    
    logger.info(f"提取了 {len(matched_entities)} 个符合条件的关键词实体")
    return matched_entities, entity_id_map

def extract_entities_by_ids(nodes_data, entity_ids, entity_types=None):
    """
    根据实体ID列表从节点数据中提取实体
    
    Parameters:
    - nodes_data: 节点数据列表
    - entity_ids: 实体ID列表
    - entity_types: 实体类型列表（可选）
    
    Returns:
    - 符合条件的实体列表和ID映射
    """
    logger.info(f"根据ID提取实体: ID数量={len(entity_ids)}, 实体类型={entity_types}")
    
    # 确保nodes_data是列表
    if not isinstance(nodes_data, list):
        logger.error(f"意外的数据类型: {type(nodes_data)}")
        return [], {}
    
    # 创建ID集合用于快速查找
    entity_id_set = set(entity_ids)
    
    matched_entities = []
    entity_id_map = {}  # 用于映射biokdeid到新的顺序ID
    
    # 遍历所有节点
    for node in tqdm(nodes_data, desc="处理实体", total=len(nodes_data)):
        # 确保节点是字典
        if not isinstance(node, dict):
            continue
        
        # 获取实体ID和类型
        entity_id = node.get("biokdeid", "")
        entity_type = node.get("type", "")
        
        # 检查ID是否在指定列表中
        id_match = entity_id in entity_id_set
        
        # 检查实体类型是否匹配（如果指定了类型）
        type_match = not entity_types or entity_type in entity_types
        
        # 如果ID和类型都匹配
        if id_match and type_match:
            # 清理名称中的非ASCII字符，但保留希腊字母
            cleaned_official_name = clean_non_ascii_chars(node.get("official name", ""), preserve_greek=True)
            cleaned_common_name = clean_non_ascii_chars(node.get("common name", ""), preserve_greek=True)
            
            # 创建实体记录
            entity = {
                "Original ID": entity_id,
                "Name": cleaned_common_name if cleaned_common_name else cleaned_official_name,
                "Official_Name": cleaned_official_name,
                "Common_Name": cleaned_common_name,
                "Type": entity_type,
                "Subtype": node.get("subtype", ""),
                "External ID": node.get("id", ""),
                "Species": ", ".join(map(str, node.get("species", []))) if node.get("species") else "",
                "Is Keyword": entity_id in entity_id_set  # 标记关键词实体
            }
            
            matched_entities.append(entity)
    
    # 为匹配的实体分配顺序ID
    for i, entity in enumerate(matched_entities):
        entity["ID"] = i + 1
        entity_id_map[entity["Original ID"]] = entity["ID"]
    
    logger.info(f"提取了 {len(matched_entities)} 个符合条件的实体")
    return matched_entities, entity_id_map

def extract_focal_entities(nodes_data, drug_keywords=None, disease_keywords=None, exact_match=False):
    """
    提取焦点实体（药物和疾病）
    
    Parameters:
    - nodes_data: 节点数据列表
    - drug_keywords: 药物关键词列表
    - disease_keywords: 疾病关键词列表
    - exact_match: 是否使用精确匹配
    
    Returns:
    - 字典，包含药物和疾病实体及其ID映射
    """
    logger.info(f"提取焦点实体: 药物关键词={drug_keywords}, 疾病关键词={disease_keywords}")
    
    # 提取药物实体
    drug_entities = []
    drug_id_map = {}
    
    if drug_keywords:
        drug_entities, drug_id_map = extract_keyword_entities(
            nodes_data, 
            keywords=drug_keywords,
            entity_types=["Chemical"],
            exact_match=exact_match
        )
        logger.info(f"提取了 {len(drug_entities)} 种药物")
    
    # 提取疾病实体
    disease_entities = []
    disease_id_map = {}
    
    if disease_keywords:
        disease_entities, disease_id_map = extract_keyword_entities(
            nodes_data, 
            keywords=disease_keywords,
            entity_types=["Disease"],
            exact_match=exact_match
        )
        logger.info(f"提取了 {len(disease_entities)} 种疾病")
    
    # 合并ID映射
    entity_id_map = {}
    entity_id_map.update(drug_id_map)
    entity_id_map.update(disease_id_map)
    
    # 合并实体列表
    all_entities = drug_entities + disease_entities
    
    # 创建焦点实体结构
    focal_entities = {
        'drug': [entity["Original ID"] for entity in drug_entities],
        'disease': [entity["Original ID"] for entity in disease_entities]
    }
    
    result = {
        'all_entities': all_entities,
        'entity_id_map': entity_id_map,
        'focal_entities': focal_entities,
        'drug_entities': drug_entities,
        'disease_entities': disease_entities
    }
    
    return result

def get_entity_by_keyword(entities_df, keyword, exact_match=False):
    """
    根据关键词从实体DataFrame中查找实体
    
    Parameters:
    - entities_df: 实体DataFrame
    - keyword: 关键词
    - exact_match: 是否精确匹配
    
    Returns:
    - 匹配的实体（如有多个，则返回第一个）
    """
    keyword = keyword.lower()
    
    if exact_match:
        # 精确匹配
        mask = (entities_df['Name'].str.lower() == keyword) | \
               (entities_df['Official_Name'].str.lower() == keyword) | \
               (entities_df['Common_Name'].str.lower() == keyword)
    else:
        # 部分匹配
        mask = (entities_df['Name'].str.lower().str.contains(keyword, na=False)) | \
               (entities_df['Official_Name'].str.lower().str.contains(keyword, na=False)) | \
               (entities_df['Common_Name'].str.lower().str.contains(keyword, na=False))
    
    matched_entities = entities_df[mask]
    
    if len(matched_entities) > 0:
        logger.info(f"找到 {len(matched_entities)} 个匹配 '{keyword}' 的实体")
        return matched_entities.iloc[0]
    else:
        logger.warning(f"未找到匹配 '{keyword}' 的实体")
        return None

def get_entity_by_id(entities_df, entity_id, id_column='ID'):
    """
    根据ID从实体DataFrame中查找实体
    
    Parameters:
    - entities_df: 实体DataFrame
    - entity_id: 实体ID
    - id_column: ID列名（默认为'ID'）
    
    Returns:
    - 匹配的实体或None
    """
    matched_entities = entities_df[entities_df[id_column] == entity_id]
    
    if len(matched_entities) > 0:
        return matched_entities.iloc[0]
    else:
        logger.warning(f"未找到ID为 '{entity_id}' 的实体")
        return None