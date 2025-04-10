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
import multiprocessing as mp
import traceback as tb  # 重命名为tb避免作用域冲突
from tqdm import tqdm
from datetime import datetime

# 尝试导入rapidjson，如果不可用则使用标准json
try:
    import rapidjson
except ImportError:
    print("警告: 未找到rapidjson，使用标准json库。这可能导致CJK字符处理问题。")
    import json as rapidjson

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from config.settings import PATHS, DATA_LOADING
from core.data_loader import load_json_file, enable_large_dataset_processing
from core.entity_extractor import extract_entities_by_ids
from core.relation_extractor import extract_relations_with_entities, update_relation_entity_ids, add_entity_names_to_relations
from utils.file_utils import check_directories, save_entities_to_csv, save_relations_to_csv

logger = logging.getLogger(__name__)

# 设置批处理大小
BATCH_SIZE = 100000  # 积累100,000个对象再处理

def clean_non_ascii_chars(text, preserve_greek=False):
    """
    清理文本中的非ASCII字符
    
    Parameters:
    - text: 输入文本
    - preserve_greek: 是否保留希腊字母
    
    Returns:
    - 清理后的文本
    """
    if not text:
        return ""
    
    # 如果保留希腊字母，则使用更复杂的替换逻辑
    if preserve_greek:
        # 希腊字母Unicode范围
        greek_lower_start = 0x03B1  # α (alpha)
        greek_lower_end = 0x03C9    # ω (omega)
        greek_upper_start = 0x0391  # Α (Alpha)
        greek_upper_end = 0x03A9    # Ω (Omega)
        
        # 创建一个过滤函数
        def is_acceptable_char(c):
            code = ord(c)
            is_ascii = code < 128
            is_greek_lower = greek_lower_start <= code <= greek_lower_end
            is_greek_upper = greek_upper_start <= code <= greek_upper_end
            return is_ascii or is_greek_lower or is_greek_upper
        
        # 过滤字符
        return ''.join(c for c in text if is_acceptable_char(c))
    else:
        # 否则仅保留ASCII字符
        return ''.join(c for c in text if ord(c) < 128)

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
    
    # 预过滤节点数据，只保留指定类型的实体
    if entity_types:
        filtered_nodes = []
        for node in nodes_data:
            if isinstance(node, dict) and node.get("type") in entity_types:
                filtered_nodes.append(node)
        nodes_data = filtered_nodes
        print(f"根据实体类型过滤后剩余 {len(nodes_data)} 个节点")
    
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
        
        # 检查实体类型是否匹配 - 由于已经预过滤，这个检查可能是多余的，但保留以确保安全
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

def check_pubmed_file_format(file_path):
    """检查PubMed文件的格式，以便更好地理解数据结构"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # 读取前几行
            first_lines = []
            for _ in range(5):
                try:
                    line = next(f).strip()
                    if line:
                        first_lines.append(line)
                except StopIteration:
                    break
            
        print(f"PubMed文件格式检查 ({file_path}):")
        for i, line in enumerate(first_lines):
            max_display = 100
            if len(line) > max_display:
                print(f"第{i+1}行 (长度 {len(line)}): {line[:max_display]}...")
            else:
                print(f"第{i+1}行: {line}")
            
        # 尝试解析完整的第一个对象
        try:
            # 如果文件是JSON数组，尝试读取第一个完整对象
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # 读取前10000个字符来尝试解析
                
                # 如果文件以'['开头，则寻找第一个完整的JSON对象
                if content.strip().startswith('['):
                    # 找到第一个'{'
                    obj_start = content.find('{')
                    if obj_start >= 0:
                        # 寻找匹配的'}'
                        depth = 0
                        for i, char in enumerate(content[obj_start:]):
                            if char == '{':
                                depth += 1
                            elif char == '}':
                                depth -= 1
                                if depth == 0:
                                    # 找到了完整的对象
                                    obj_end = obj_start + i + 1
                                    json_str = content[obj_start:obj_end]
                                    json_obj = json.loads(json_str)
                                    
                                    print("\n文件格式分析:")
                                    print(f"对象类型: {type(json_obj)}")
                                    print(f"对象键: {list(json_obj.keys()) if isinstance(json_obj, dict) else 'Not a dictionary'}")
                                    
                                    # 检查README中描述的格式
                                    if isinstance(json_obj, dict):
                                        if 'id' in json_obj and 'list' in json_obj:
                                            print("\n发现符合README描述的结构:")
                                            print(f"id: {json_obj['id']}")
                                            print(f"list类型: {type(json_obj['list'])}")
                                            if isinstance(json_obj['list'], list) and json_obj['list']:
                                                print(f"list第一项: {json_obj['list'][0]}")
                                        else:
                                            print("\n未找到匹配README描述的结构")
                                            print(f"实际键: {list(json_obj.keys())}")
                                    break
        except Exception as e:
            print(f"无法解析JSON: {e}")
    except Exception as e:
        print(f"检查文件格式时出错: {e}")

def process_found_objects(objects_list, focal_ids, focal_entities, relation_schema, extracted_relations, entity_connections):
    """处理找到的JSON对象并提取关系"""
    relation_count = 0
    
    # 创建焦点ID集合以加速查找
    focal_ids_set = set(focal_ids)
    
    for obj_text in objects_list:
        try:
            # 解析JSON对象
            relation_obj = json.loads(obj_text)
            
            # 检查是否符合预期结构
            if not isinstance(relation_obj, dict) or 'id' not in relation_obj or 'list' not in relation_obj:
                continue
            
            # 解析关系ID
            rel_id_parts = relation_obj['id'].split('.')
            if len(rel_id_parts) < 3:
                continue
            
            source_id = rel_id_parts[0]
            target_id = rel_id_parts[1]
            relation_type = rel_id_parts[2]
            
            # 检查是否与焦点实体相关
            if source_id in focal_ids_set or target_id in focal_ids_set:
                # 获取关系类型名称
                relation_type_name = relation_schema.get(relation_type)
                
                # 从list中提取关系数据
                for rel_data in relation_obj.get('list', []):
                    if isinstance(rel_data, list) and len(rel_data) >= 3:
                        score = rel_data[0]
                        document_id = rel_data[1]
                        probability = rel_data[2]
                        
                        # 创建关系记录
                        relation = {
                            "Original Relation ID": relation_obj['id'],
                            "Type": relation_type_name if relation_type_name else f"Unknown_{relation_type}",
                            "Original Type ID": relation_type,
                            "Original Source ID": source_id,
                            "Original Target ID": target_id,
                            "Source": "PubMed",
                            "Score": score,
                            "Document ID": document_id,
                            "Probability": probability
                        }
                        
                        # 添加到提取的关系
                        extracted_relations.append(relation)
                        relation_count += 1
                        
                        # 跟踪实体连接
                        # 对于药物
                        for drug_id in focal_entities.get('drug', []):
                            if source_id == drug_id and target_id != drug_id:
                                entity_connections['drug'][drug_id].add(target_id)
                            elif target_id == drug_id and source_id != drug_id:
                                entity_connections['drug'][drug_id].add(source_id)
                        
                        # 对于疾病
                        for disease_id in focal_entities.get('disease', []):
                            if source_id == disease_id and target_id != disease_id:
                                entity_connections['disease'][disease_id].add(target_id)
                            elif target_id == disease_id and source_id != disease_id:
                                entity_connections['disease'][disease_id].add(source_id)
        
        except json.JSONDecodeError:
            pass  # 忽略解析错误，继续处理下一个对象
        except Exception as e:
            pass  # 忽略所有错误，提高处理速度
    
    if relation_count > 0:
        print(f"在当前批次中找到 {relation_count} 个关系")

def process_pubmed_relations(file_path, focal_ids, focal_entities, relation_schema):
    """
    使用字符级解析处理PubMed关系数据，积累100,000个对象再处理
    """
    print(f"使用大批量处理PubMed关系（每批 {BATCH_SIZE:,} 个对象）...")
    
    extracted_relations = []
    entity_connections = {
        'drug': {drug_id: set() for drug_id in focal_entities.get('drug', [])},
        'disease': {disease_id: set() for disease_id in focal_entities.get('disease', [])}
    }
    
    try:
        # 获取文件大小以估计总进度
        file_size = os.path.getsize(file_path)
        start_time = datetime.now()
        print(f"文件大小: {file_size/1024/1024:.2f} MB")
        
        # 读取文件的前100个字符来检测格式
        with open(file_path, 'r', encoding='utf-8') as f:
            prefix = f.read(100)
            
        # 检查文件是否是JSON数组格式
        if not prefix.lstrip().startswith('['):
            print("警告: PubMed文件不是JSON数组格式，期望文件以'['开头")
            return [], {}
        
        # 使用字符级解析
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过文件开头的'['
            if f.read(1) != '[':
                f.seek(0)  # 重置到文件开头
            
            # 跟踪状态变量
            objects_found = []
            bracket_depth = 0
            in_object = False
            in_string = False
            escape_next = False
            start_pos = 0
            
            total_objects = 0
            total_processed_bytes = 0
            last_report_time = datetime.now()
            last_report_objects = 0
            
            # 每次读取5MB数据进行处理
            chunk_size = 5 * 1024 * 1024
            
            while True:
                # 读取一块数据
                chunk = f.read(chunk_size)
                if not chunk:  # 文件结束
                    break
                
                total_processed_bytes += len(chunk)
                
                # 逐字符分析，识别完整的JSON对象
                for i, char in enumerate(chunk):
                    # 处理字符串中的字符
                    if in_string:
                        if escape_next:
                            escape_next = False
                        elif char == '\\':
                            escape_next = True
                        elif char == '"':
                            in_string = False
                        continue
                    
                    # 处理非字符串内容
                    if char == '"':
                        in_string = True
                    elif char == '{':
                        if bracket_depth == 0:
                            # 这是一个新对象的开始
                            start_pos = total_processed_bytes - len(chunk) + i
                            in_object = True
                        bracket_depth += 1
                    elif char == '}':
                        bracket_depth -= 1
                        if bracket_depth == 0 and in_object:
                            # 找到了一个完整的JSON对象
                            end_pos = total_processed_bytes - len(chunk) + i + 1
                            
                            # 读取对象内容
                            f.seek(start_pos)
                            obj_text = f.read(end_pos - start_pos)
                            f.seek(total_processed_bytes)  # 回到当前位置
                            
                            objects_found.append(obj_text)
                            total_objects += 1
                            in_object = False
                            
                            # 定期报告进度
                            current_time = datetime.now()
                            time_diff = (current_time - last_report_time).total_seconds()
                            
                            if time_diff >= 5:  # 每5秒报告一次进度
                                elapsed_time = (current_time - start_time).total_seconds()
                                percent_complete = total_processed_bytes / file_size * 100
                                objects_per_second = (total_objects - last_report_objects) / time_diff
                                
                                print(f"进度: {percent_complete:.1f}% | 已处理: {total_objects:,} 个对象 | "
                                      f"速率: {objects_per_second:.1f} 对象/秒 | "
                                      f"已用时间: {elapsed_time:.0f}秒")
                                
                                last_report_time = current_time
                                last_report_objects = total_objects
                            
                            # 为了避免内存问题，当积累了100,000个对象时处理它们
                            if len(objects_found) >= BATCH_SIZE:
                                print(f"批量处理 {len(objects_found):,} 个对象...")
                                process_found_objects(objects_found, focal_ids, focal_entities, 
                                                      relation_schema, extracted_relations, entity_connections)
                                objects_found = []  # 清空对象列表
                                gc.collect()  # 强制垃圾回收
                
                # 如果块结束时我们正在处理一个对象，需要调整文件指针
                if in_object:
                    # 回退到起始位置
                    f.seek(start_pos)
                    total_processed_bytes = start_pos
                    # 重置状态
                    in_object = False
                    bracket_depth = 0
                    in_string = False
                    escape_next = False
        
            # 处理剩余的对象
            if objects_found:
                print(f"处理剩余的 {len(objects_found):,} 个对象...")
                process_found_objects(objects_found, focal_ids, focal_entities, relation_schema, 
                                      extracted_relations, entity_connections)
        
        # 计算和报告总处理时间
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        print(f"PubMed处理完成，找到 {len(extracted_relations):,} 个关系")
        print(f"总处理时间: {elapsed_time:.1f}秒 | "
              f"平均速率: {total_objects/elapsed_time:.1f} 对象/秒")
    
    except Exception as e:
        print(f"处理PubMed文件时出错: {e}")
        print(tb.format_exc())
    
    return extracted_relations, entity_connections

def efficient_pubmed_processing(file_path, focal_ids, focal_entities, relation_schema):
    """
    高效处理PubMed关系数据 - 使用流式解析和大批量处理
    """
    print(f"使用高效流式处理方法处理PubMed关系（每批 {BATCH_SIZE:,} 个对象）...")
    
    extracted_relations = []
    entity_connections = {
        'drug': {drug_id: set() for drug_id in focal_entities.get('drug', [])},
        'disease': {disease_id: set() for disease_id in focal_entities.get('disease', [])}
    }
    
    # 创建焦点ID集合用于快速查找
    focal_ids_set = set(focal_ids)
    print(f"Focal IDs数量: {len(focal_ids_set)}")
    
    # 状态跟踪变量
    relation_count = 0
    object_count = 0
    start_time = datetime.now()
    last_report_time = start_time
    last_report_objects = 0
    
    # 创建一个缓冲区来存储对象，直到积累到足够多
    objects_buffer = []
    
    try:
        # 获取文件大小以计算进度
        file_size = os.path.getsize(file_path)
        print(f"文件大小: {file_size/1024/1024:.2f} MB")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过文件开头的'['
            first_char = f.read(1)
            if first_char != '[':
                f.seek(0)  # 重置到文件开头
            
            # 读取并处理文件
            json_buffer = ""
            in_object = False
            bracket_depth = 0
            in_string = False
            escape_next = False
            total_processed_bytes = 1  # 初始为1，因为已经读取了第一个字符
            
            # 设置读取缓冲区大小 - 每次读取5MB
            chunk_size = 5 * 1024 * 1024
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:  # 文件结束
                    break
                
                total_processed_bytes += len(chunk)
                
                # 处理当前块
                i = 0
                while i < len(chunk):
                    char = chunk[i]
                    i += 1
                    
                    # 处理字符串中的字符
                    if in_string:
                        json_buffer += char
                        if escape_next:
                            escape_next = False
                        elif char == '\\':
                            escape_next = True
                        elif char == '"':
                            in_string = False
                        continue
                    
                    # 处理非字符串内容
                    json_buffer += char
                    
                    if char == '"':
                        in_string = True
                    elif char == '{':
                        if bracket_depth == 0:
                            in_object = True
                        bracket_depth += 1
                    elif char == '}':
                        bracket_depth -= 1
                        if bracket_depth == 0 and in_object:
                            # 完成了一个JSON对象
                            try:
                                # 尝试解析对象
                                relation_obj = json.loads(json_buffer)
                                object_count += 1
                                
                                # 处理符合条件的关系
                                if (isinstance(relation_obj, dict) and 
                                    'id' in relation_obj and 'list' in relation_obj):
                                    
                                    rel_id_parts = relation_obj['id'].split('.')
                                    if len(rel_id_parts) >= 3:
                                        source_id = rel_id_parts[0]
                                        target_id = rel_id_parts[1]
                                        rel_type = rel_id_parts[2]
                                        
                                        # 检查是否与焦点实体相关
                                        if source_id in focal_ids_set or target_id in focal_ids_set:
                                            # 添加到对象缓冲区
                                            objects_buffer.append(relation_obj)
                                
                            except json.JSONDecodeError:
                                pass  # 忽略解析错误
                            except Exception:
                                pass  # 忽略所有错误以提高速度
                            
                            # 重置状态
                            json_buffer = ""
                            in_object = False
                            
                            # 报告进度
                            current_time = datetime.now()
                            time_diff = (current_time - last_report_time).total_seconds()
                            
                            # 每5秒或每1,000,000个对象报告一次进度
                            if time_diff >= 5 or (object_count - last_report_objects) >= 1000000:
                                elapsed_time = (current_time - start_time).total_seconds()
                                percent_complete = total_processed_bytes / file_size * 100
                                objects_per_second = (object_count - last_report_objects) / max(1, time_diff)
                                
                                print(f"进度: {percent_complete:.1f}% | 已处理: {object_count:,} 个对象 | "
                                      f"找到匹配: {len(objects_buffer):,} | "
                                      f"速率: {objects_per_second:.1f} 对象/秒 | "
                                      f"已用时间: {elapsed_time:.0f}秒")
                                
                                last_report_time = current_time
                                last_report_objects = object_count
                            
                            # 当积累了100,000个匹配对象时处理它们
                            if len(objects_buffer) >= BATCH_SIZE:
                                print(f"\n处理 {len(objects_buffer):,} 个匹配对象...")
                                batch_start = datetime.now()
                                
                                for obj in objects_buffer:
                                    try:
                                        source_id = obj['id'].split('.')[0]
                                        target_id = obj['id'].split('.')[1]
                                        rel_type = obj['id'].split('.')[2]
                                        rel_type_name = relation_schema.get(rel_type)
                                        
                                        # 从list中提取关系数据
                                        for rel_data in obj.get('list', []):
                                            if isinstance(rel_data, list) and len(rel_data) >= 3:
                                                # 创建关系记录
                                                relation = {
                                                    "Original Relation ID": obj['id'],
                                                    "Type": rel_type_name if rel_type_name else f"Unknown_{rel_type}",
                                                    "Original Type ID": rel_type,
                                                    "Original Source ID": source_id,
                                                    "Original Target ID": target_id,
                                                    "Source": "PubMed",
                                                    "Score": rel_data[0],
                                                    "Document ID": rel_data[1],
                                                    "Probability": rel_data[2]
                                                }
                                                
                                                extracted_relations.append(relation)
                                                relation_count += 1
                                                
                                                # 更新实体连接
                                                for drug_id in focal_entities.get('drug', []):
                                                    if source_id == drug_id and target_id != drug_id:
                                                        entity_connections['drug'][drug_id].add(target_id)
                                                    elif target_id == drug_id and source_id != drug_id:
                                                        entity_connections['drug'][drug_id].add(source_id)
                                                
                                                for disease_id in focal_entities.get('disease', []):
                                                    if source_id == disease_id and target_id != disease_id:
                                                        entity_connections['disease'][disease_id].add(target_id)
                                                    elif target_id == disease_id and source_id != disease_id:
                                                        entity_connections['disease'][disease_id].add(source_id)
                                    except Exception:
                                        pass  # 忽略错误，继续处理
                                
                                batch_time = (datetime.now() - batch_start).total_seconds()
                                print(f"批处理完成，找到 {relation_count:,} 个关系，用时: {batch_time:.1f}秒")
                                
                                objects_buffer = []  # 清空缓冲区
                                gc.collect()  # 强制垃圾回收
            
            # 处理剩余的对象
            if objects_buffer:
                print(f"\n处理剩余的 {len(objects_buffer):,} 个匹配对象...")
                batch_start = datetime.now()
                
                for obj in objects_buffer:
                    try:
                        source_id = obj['id'].split('.')[0]
                        target_id = obj['id'].split('.')[1]
                        rel_type = obj['id'].split('.')[2]
                        rel_type_name = relation_schema.get(rel_type)
                        
                        # 从list中提取关系数据
                        for rel_data in obj.get('list', []):
                            if isinstance(rel_data, list) and len(rel_data) >= 3:
                                # 创建关系记录
                                relation = {
                                    "Original Relation ID": obj['id'],
                                    "Type": rel_type_name if rel_type_name else f"Unknown_{rel_type}",
                                    "Original Type ID": rel_type,
                                    "Original Source ID": source_id,
                                    "Original Target ID": target_id,
                                    "Source": "PubMed",
                                    "Score": rel_data[0],
                                    "Document ID": rel_data[1],
                                    "Probability": rel_data[2]
                                }
                                
                                extracted_relations.append(relation)
                                relation_count += 1
                                
                                # 更新实体连接
                                for drug_id in focal_entities.get('drug', []):
                                    if source_id == drug_id and target_id != drug_id:
                                        entity_connections['drug'][drug_id].add(target_id)
                                    elif target_id == drug_id and source_id != drug_id:
                                        entity_connections['drug'][drug_id].add(source_id)
                                
                                for disease_id in focal_entities.get('disease', []):
                                    if source_id == disease_id and target_id != disease_id:
                                        entity_connections['disease'][disease_id].add(target_id)
                                    elif target_id == disease_id and source_id != disease_id:
                                        entity_connections['disease'][disease_id].add(source_id)
                    except Exception:
                        pass  # 忽略错误，继续处理
                
                batch_time = (datetime.now() - batch_start).total_seconds()
                print(f"批处理完成，找到 {relation_count:,} 个关系，用时: {batch_time:.1f}秒")
            
            # 计算和报告总处理时间
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"\nPubMed处理完成，处理了 {object_count:,} 个对象，找到 {relation_count:,} 个关系")
            print(f"总处理时间: {total_time:.1f}秒 | "
                  f"平均速率: {object_count/total_time:.1f} 对象/秒")
    
    except Exception as e:
        print(f"处理PubMed文件时出错: {e}")
        print(tb.format_exc())
    
    return extracted_relations, entity_connections

def simple_pubmed_processing(file_path, focal_ids, focal_entities, relation_schema):
    """
    简单的PubMed关系处理方法，使用行读取和大批处理
    """
    print(f"尝试使用简单行处理方法处理PubMed关系（每批 {BATCH_SIZE:,} 个对象）...")
    
    extracted_relations = []
    entity_connections = {
        'drug': {drug_id: set() for drug_id in focal_entities.get('drug', [])},
        'disease': {disease_id: set() for disease_id in focal_entities.get('disease', [])}
    }
    
    # 创建焦点ID集合用于快速查找
    focal_ids_set = set(focal_ids)
    
    try:
        # 获取文件大小以计算进度
        file_size = os.path.getsize(file_path)
        start_time = datetime.now()
        
        # 尝试按照小段读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过文件开头的'['
            f.readline()
            
            # 处理每个对象
            buffer = ""
            depth = 0
            in_object = False
            object_count = 0
            relation_count = 0
            last_report_time = start_time
            last_report_objects = 0
            
            # 创建对象缓冲区
            objects_buffer = []
            
            for line in f:
                line = line.strip()
                if not line or line == ']':
                    continue
                
                # 如果行以'{'开头，可能是新对象的开始
                if line.startswith('{'):
                    buffer = line
                    depth = line.count('{') - line.count('}')
                    in_object = True
                    continue
                
                # 如果在处理一个对象，继续累积内容
                if in_object:
                    buffer += line
                    depth += line.count('{') - line.count('}')
                    
                    # 如果深度回到0，对象可能已经完成
                    if depth <= 0:
                        # 移除末尾的逗号
                        if buffer.endswith(','):
                            buffer = buffer[:-1]
                        
                        try:
                            # 尝试解析对象
                            obj = json.loads(buffer)
                            object_count += 1
                            
                            # 检查对象结构
                            if isinstance(obj, dict) and 'id' in obj and 'list' in obj:
                                # 解析关系ID
                                parts = obj['id'].split('.')
                                if len(parts) >= 3:
                                    source_id = parts[0]
                                    target_id = parts[1]
                                    
                                    # 检查是否与焦点实体相关
                                    if source_id in focal_ids_set or target_id in focal_ids_set:
                                        # 将对象添加到缓冲区
                                        objects_buffer.append(obj)
                            
                            # 报告进度
                            current_time = datetime.now()
                            time_diff = (current_time - last_report_time).total_seconds()
                            
                            if time_diff >= 5 or (object_count - last_report_objects) >= 1000000:
                                elapsed_time = (current_time - start_time).total_seconds()
                                tell_position = f.tell()
                                percent_complete = tell_position / file_size * 100
                                objects_per_second = (object_count - last_report_objects) / max(1, time_diff)
                                
                                print(f"进度: {percent_complete:.1f}% | 已处理: {object_count:,} 个对象 | "
                                      f"找到匹配: {len(objects_buffer):,} | "
                                      f"速率: {objects_per_second:.1f} 对象/秒 | "
                                      f"已用时间: {elapsed_time:.0f}秒")
                                
                                last_report_time = current_time
                                last_report_objects = object_count
                            
                            # 当积累了100,000个匹配对象时处理它们
                            if len(objects_buffer) >= BATCH_SIZE:
                                print(f"\n处理 {len(objects_buffer):,} 个匹配对象...")
                                batch_start = datetime.now()
                                
                                batch_relations = 0
                                for obj in objects_buffer:
                                    try:
                                        source_id = obj['id'].split('.')[0]
                                        target_id = obj['id'].split('.')[1]
                                        rel_type = obj['id'].split('.')[2]
                                        rel_type_name = relation_schema.get(rel_type)
                                        
                                        # 从list中提取关系数据
                                        for rel_data in obj.get('list', []):
                                            if isinstance(rel_data, list) and len(rel_data) >= 3:
                                                # 创建关系记录
                                                relation = {
                                                    "Original Relation ID": obj['id'],
                                                    "Type": rel_type_name if rel_type_name else f"Unknown_{rel_type}",
                                                    "Original Type ID": rel_type,
                                                    "Original Source ID": source_id,
                                                    "Original Target ID": target_id,
                                                    "Source": "PubMed",
                                                    "Score": rel_data[0],
                                                    "Document ID": rel_data[1],
                                                    "Probability": rel_data[2]
                                                }
                                                
                                                extracted_relations.append(relation)
                                                relation_count += 1
                                                batch_relations += 1
                                                
                                                # 更新实体连接
                                                for drug_id in focal_entities.get('drug', []):
                                                    if source_id == drug_id and target_id != drug_id:
                                                        entity_connections['drug'][drug_id].add(target_id)
                                                    elif target_id == drug_id and source_id != drug_id:
                                                        entity_connections['drug'][drug_id].add(source_id)
                                                
                                                for disease_id in focal_entities.get('disease', []):
                                                    if source_id == disease_id and target_id != disease_id:
                                                        entity_connections['disease'][disease_id].add(target_id)
                                                    elif target_id == disease_id and source_id != disease_id:
                                                        entity_connections['disease'][disease_id].add(source_id)
                                    except Exception:
                                        pass  # 忽略错误，继续处理
                                
                                batch_time = (datetime.now() - batch_start).total_seconds()
                                print(f"批处理完成，找到 {batch_relations:,} 个关系，总计: {relation_count:,}，用时: {batch_time:.1f}秒")
                                
                                objects_buffer = []  # 清空缓冲区
                                gc.collect()  # 强制垃圾回收
                                
                        except json.JSONDecodeError:
                            # 忽略解析错误，继续处理下一个对象
                            pass
                        
                        # 重置状态
                        buffer = ""
                        in_object = False
                        depth = 0
            
            # 处理剩余的对象
            if objects_buffer:
                print(f"\n处理剩余的 {len(objects_buffer):,} 个匹配对象...")
                batch_start = datetime.now()
                
                batch_relations = 0
                for obj in objects_buffer:
                    try:
                        source_id = obj['id'].split('.')[0]
                        target_id = obj['id'].split('.')[1]
                        rel_type = obj['id'].split('.')[2]
                        rel_type_name = relation_schema.get(rel_type)
                        
                        # 从list中提取关系数据
                        for rel_data in obj.get('list', []):
                            if isinstance(rel_data, list) and len(rel_data) >= 3:
                                # 创建关系记录
                                relation = {
                                    "Original Relation ID": obj['id'],
                                    "Type": rel_type_name if rel_type_name else f"Unknown_{rel_type}",
                                    "Original Type ID": rel_type,
                                    "Original Source ID": source_id,
                                    "Original Target ID": target_id,
                                    "Source": "PubMed",
                                    "Score": rel_data[0],
                                    "Document ID": rel_data[1],
                                    "Probability": rel_data[2]
                                }
                                
                                extracted_relations.append(relation)
                                relation_count += 1
                                batch_relations += 1
                                
                                # 更新实体连接
                                for drug_id in focal_entities.get('drug', []):
                                    if source_id == drug_id and target_id != drug_id:
                                        entity_connections['drug'][drug_id].add(target_id)
                                    elif target_id == drug_id and source_id != drug_id:
                                        entity_connections['drug'][drug_id].add(source_id)
                                
                                for disease_id in focal_entities.get('disease', []):
                                    if source_id == disease_id and target_id != disease_id:
                                        entity_connections['disease'][disease_id].add(target_id)
                                    elif target_id == disease_id and source_id != disease_id:
                                        entity_connections['disease'][disease_id].add(source_id)
                    except Exception:
                        pass  # 忽略错误，继续处理
                
                batch_time = (datetime.now() - batch_start).total_seconds()
                print(f"批处理完成，找到 {batch_relations:,} 个关系，总计: {relation_count:,}，用时: {batch_time:.1f}秒")
        
        # 计算和报告总处理时间
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n简单处理完成，处理了 {object_count:,} 个对象，找到 {relation_count:,} 个关系")
        print(f"总处理时间: {total_time:.1f}秒 | "
              f"平均速率: {object_count/total_time:.1f} 对象/秒")
    
    except Exception as e:
        print(f"简单处理时出错: {e}")
        print(tb.format_exc())
    
    return extracted_relations, entity_connections

def parse_arguments():
    """解析命令行参数"""
    global BATCH_SIZE
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
    parser.add_argument('--process_count', type=int, default=max(4, mp.cpu_count() // 2), help='并行处理的进程数')
    parser.add_argument('--low_memory', action='store_true', help='启用低内存模式')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='PubMed处理的批量大小')
    
    args = parser.parse_args()
    
    # 设置全局批处理大小

    if args.batch_size > 0:
        BATCH_SIZE = args.batch_size
    
    # 添加调试输出
    print("解析后的参数:")
    print("数据目录:", args.data_dir)
    print("输出目录:", args.output_dir)
    print("药物关键词:", args.drug_keywords)
    print("疾病关键词:", args.disease_keywords)
    print("精确匹配:", args.exact_match)
    print("使用进程数:", args.process_count)
    print("块大小:", args.chunk_size)
    print("批处理大小:", BATCH_SIZE)
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
                # 预过滤节点数据，只保留可能是药物的实体
                chemical_nodes = [node for node in nodes_data if isinstance(node, dict) and node.get("type") == "Chemical"]
                print(f"预过滤后的化学实体节点数: {len(chemical_nodes)}")
                
                # 处理多个药物关键词，逐个提取
                for drug_keyword in args.drug_keywords:
                    logger.info(f"提取药物: {drug_keyword}")
                    print(f"正在提取药物: {drug_keyword}")
                    drug_entities_single, drug_id_map_single = extract_keyword_entities(
                        chemical_nodes,  # 使用预过滤的节点
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
                # 预过滤节点数据，只保留可能是疾病的实体
                disease_nodes = [node for node in nodes_data if isinstance(node, dict) and node.get("type") == "Disease"]
                print(f"预过滤后的疾病实体节点数: {len(disease_nodes)}")
                
                for disease_keyword in args.disease_keywords:
                    logger.info(f"提取疾病: {disease_keyword}")
                    print(f"正在提取疾病: {disease_keyword}")
                    disease_entities_single, disease_id_map_single = extract_keyword_entities(
                        disease_nodes,  # 使用预过滤的节点
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
                relation_schema_raw = json.load(f)
                
            # 转换关系模式为字符串键值映射（intRep -> relType）
            relation_schema = {}
            
            # 根据文件内容，这是一个对象列表，每个对象都有relType和intRep
            if isinstance(relation_schema_raw, list):
                for item in relation_schema_raw:
                    if isinstance(item, dict) and "intRep" in item and "relType" in item:
                        relation_schema[item["intRep"]] = item["relType"]
            
            print(f"成功加载关系模式，包含 {len(relation_schema)} 种关系类型")
        except Exception as e:
            print(f"加载关系模式失败: {e}")
            print(tb.format_exc())
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
        
        # 提取PubMed关系 - 使用高性能方法
        logger.info("提取PubMed关系...")
        pubmed_file = os.path.join(args.data_dir, "PubMedList.json")
        print(f"尝试加载PubMed文件: {pubmed_file}")
        print(f"文件存在: {os.path.exists(pubmed_file)}")
        
        # 首先检查文件格式以了解其结构
        check_pubmed_file_format(pubmed_file)

        try:
            # 创建一个更加聚焦的实体结构
            simple_focal_entities = {
                'drug': [entity["Original ID"] for entity in drug_entities] if 'drug_entities' in locals() else [],
                'disease': [entity["Original ID"] for entity in disease_entities] if 'disease_entities' in locals() else []
            }
            
            # 尝试不同的处理方法，按照推荐的顺序
            # 1. 首先尝试高效流式处理
            pubmed_relations = []
            entity_connections = {}
            
            print("尝试方法1: 高效流式处理")
            pubmed_relations, entity_connections = efficient_pubmed_processing(
                pubmed_file,
                focal_ids,
                simple_focal_entities,
                relation_schema
            )
            
            # 2. 如果没有找到关系，尝试简单行处理方法
            if not pubmed_relations:
                print("\n方法1未找到关系，尝试方法2: 简单行处理")
                pubmed_relations, entity_connections = simple_pubmed_processing(
                    pubmed_file,
                    focal_ids,
                    simple_focal_entities,
                    relation_schema
                )
            
            # 3. 如果仍然没有找到关系，尝试原始处理方法
            if not pubmed_relations:
                print("\n方法2未找到关系，尝试方法3: 原始字符级处理")
                pubmed_relations, entity_connections = process_pubmed_relations(
                    pubmed_file,
                    focal_ids,
                    simple_focal_entities,
                    relation_schema
                )
            
            logger.info(f"提取了 {len(pubmed_relations)} 条PubMed关系")
            print(f"提取了 {len(pubmed_relations)} 条PubMed关系")
        except Exception as e:
            print(f"处理PubMed关系时出错: {e}")
            print(tb.format_exc())
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
        related_ids = set()
        for relation in all_relations:
            related_ids.add(relation.get("Original Source ID", ""))
            related_ids.add(relation.get("Original Target ID", ""))
        
        # 移除焦点实体ID和空值
        related_ids = {id for id in related_ids if id and id not in focal_ids}
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
        print(tb.format_exc())  # 使用tb而不是traceback
        logger.error(f"数据提取过程中发生错误: {e}")
        logger.error(tb.format_exc())  # 使用tb而不是traceback
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
        # 设置进程启动方法
        if sys.platform.startswith('win'):
            # Windows下使用'spawn'方法
            mp.set_start_method('spawn', force=True)
        else:
            # Unix系统使用'fork'方法
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # 可能已经设置
        
        # 顶层输出只打印一次
        print("数据提取脚本启动...")
        sys.exit(main())
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print(tb.format_exc())  # 使用tb而不是traceback
        sys.exit(1)