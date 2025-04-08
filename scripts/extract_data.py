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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import io
import ijson  # 添加ijson库用于高效解析大型JSON文件
import re

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

# 设置全局常量
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB 块大小
MAX_WORKERS = max(1, mp.cpu_count() - 1)  # 使用所有CPU核心数减1

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
    
    # 预筛选实体类型 - 这将大大减少需要处理的节点数量
    filtered_nodes = []
    if entity_types:
        for node in nodes_data:
            if isinstance(node, dict) and node.get("type") in entity_types:
                filtered_nodes.append(node)
    else:
        filtered_nodes = nodes_data
    
    print(f"根据实体类型过滤后剩余 {len(filtered_nodes)} 个节点")
    
    matched_entities = []
    entity_id_map = {}  # 用于映射biokdeid到新的顺序ID
    
    # 预编译正则表达式提高性能（针对非中文关键词）
    regex_patterns = {}
    if keywords and not exact_match:
        for keyword in keywords:
            if not any('\u4e00' <= char <= '\u9fff' for char in keyword):
                regex_patterns[keyword] = re.compile(re.escape(keyword.lower()))
    
    # 遍历所有节点
    total_nodes = len(filtered_nodes)
    for node in tqdm(filtered_nodes, desc="处理关键词实体", total=total_nodes):
        # 获取实体名称
        original_official_name = str(node.get("official name", ""))
        original_common_name = str(node.get("common name", ""))
        
        # 获取实体类型
        entity_type = node.get("type", "")
        
        # 检查实体类型是否匹配 - 实际上，由于我们已经过滤了节点，这个检查是多余的
        # 但保留它以防万一逻辑在其他地方改变
        type_match = not entity_types or entity_type in entity_types
        if not type_match:
            continue
        
        # 检查关键词是否匹配
        keyword_match = False
        if keywords:
            official_name_lower = None  # 懒加载
            common_name_lower = None   # 懒加载
            
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
                        if official_name_lower is None:
                            official_name_lower = original_official_name.lower()
                        if common_name_lower is None:
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
                        # 使用预编译的正则表达式进行更快的部分匹配
                        pattern = regex_patterns.get(keyword)
                        if pattern:
                            if official_name_lower is None:
                                official_name_lower = original_official_name.lower()
                            if common_name_lower is None:
                                common_name_lower = original_common_name.lower()
                                
                            if pattern.search(official_name_lower) or pattern.search(common_name_lower):
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

def process_relation_object(relation_obj, focal_ids, focal_entities, relation_schema):
    """处理单个关系对象并提取关系 - 作为worker函数用于并行处理"""
    extracted_relations = []
    entity_connections = {
        'drug': {drug_id: set() for drug_id in focal_entities.get('drug', [])},
        'disease': {disease_id: set() for disease_id in focal_entities.get('disease', [])}
    }
    
    try:
        # 检查是否符合预期结构
        if not isinstance(relation_obj, dict) or 'id' not in relation_obj or 'list' not in relation_obj:
            return [], {}
        
        # 解析关系ID
        rel_id_parts = relation_obj['id'].split('.')
        if len(rel_id_parts) < 3:
            return [], {}
        
        source_id = rel_id_parts[0]
        target_id = rel_id_parts[1]
        relation_type = rel_id_parts[2]
        
        # 检查是否与焦点实体相关
        if source_id in focal_ids or target_id in focal_ids:
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
    
    except Exception as e:
        print(f"处理对象时出错: {e}")
    
    return extracted_relations, entity_connections

def merge_entity_connections(all_connections):
    """合并多个worker返回的实体连接字典"""
    merged = {'drug': {}, 'disease': {}}
    
    # 初始化合并后的字典
    for connection_dict in all_connections:
        # 处理药物
        for drug_id in connection_dict['drug']:
            if drug_id not in merged['drug']:
                merged['drug'][drug_id] = set()
            merged['drug'][drug_id].update(connection_dict['drug'][drug_id])
        
        # 处理疾病
        for disease_id in connection_dict['disease']:
            if disease_id not in merged['disease']:
                merged['disease'][disease_id] = set()
            merged['disease'][disease_id].update(connection_dict['disease'][disease_id])
    
    return merged

def process_pubmed_chunk(chunk_data, focal_ids, focal_entities, relation_schema, chunk_index):
    """处理PubMed数据的一个块"""
    print(f"开始处理块 {chunk_index}...")
    extracted_relations = []
    entity_connections = {
        'drug': {drug_id: set() for drug_id in focal_entities.get('drug', [])},
        'disease': {disease_id: set() for disease_id in focal_entities.get('disease', [])}
    }
    
    try:
        # 处理该块中的JSON对象
        objects = []
        in_object = False
        bracket_depth = 0
        buffer = ""
        
        # 使用ijson流处理JSON对象
        # 注意：这里我们处理的是JSON数组中的项目，而不是整个文件
        items = ijson.items(io.StringIO(chunk_data), 'item')
        
        relation_count = 0
        for obj in items:
            # 处理单个JSON对象
            relations, connections = process_relation_object(obj, focal_ids, focal_entities, relation_schema)
            extracted_relations.extend(relations)
            
            # 更新连接
            for drug_id in connections['drug']:
                entity_connections['drug'][drug_id].update(connections['drug'][drug_id])
            for disease_id in connections['disease']:
                entity_connections['disease'][disease_id].update(connections['disease'][disease_id])
            
            relation_count += len(relations)
            
            # 定期报告进度
            if relation_count > 0 and relation_count % 100 == 0:
                print(f"块 {chunk_index}: 已找到 {relation_count} 个关系")
    
    except Exception as e:
        print(f"处理块 {chunk_index} 时出错: {e}")
        print(tb.format_exc())
    
    print(f"块 {chunk_index} 处理完成，找到 {len(extracted_relations)} 个关系")
    return extracted_relations, entity_connections

def process_pubmed_relations_parallel(file_path, focal_ids, focal_entities, relation_schema):
    """
    使用并行处理和ijson流解析PubMed关系数据
    """
    print("使用并行流处理解析PubMed关系...")
    
    all_relations = []
    all_connections = []
    
    try:
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        print(f"PubMed文件大小: {file_size/1024/1024:.2f} MB")
        
        # 检查文件的前100个字符来确定格式
        with open(file_path, 'r', encoding='utf-8') as f:
            prefix = f.read(100)
            
        # 检查文件是否是JSON数组格式
        if not prefix.lstrip().startswith('['):
            print("警告: PubMed文件不是JSON数组格式，期望文件以'['开头")
            return [], {}
        
        # 读取并处理整个文件
        # 首先确定如何分割文件
        chunk_size = min(CHUNK_SIZE, file_size // (MAX_WORKERS * 2) + 1)  # 确保至少有2个块
        print(f"使用 {MAX_WORKERS} 个工作进程处理，块大小: {chunk_size/1024/1024:.2f} MB")
        
        # 创建处理池
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            # 打开文件并在内容上进行迭代
            with open(file_path, 'r', encoding='utf-8') as f:
                # 跳过文件开头的'['
                first_char = f.read(1)
                if first_char != '[':
                    f.seek(0)  # 如果不是'['，重置到开头
                
                # 分块读取并处理文件
                chunk_index = 0
                while True:
                    # 读取一个数据块
                    chunk_data = f.read(chunk_size)
                    if not chunk_data:
                        break  # 文件末尾
                    
                    # 确保我们有一个完整的JSON对象 - 寻找最后一个完整的对象
                    if chunk_data[-1] != '}':
                        # 读取直到下一个完整对象的结束
                        depth = 0
                        in_string = False
                        escape_next = False
                        buffer = ""
                        
                        for char in chunk_data[::-1]:  # 从最后向前查找
                            buffer = char + buffer
                            
                            if in_string:
                                if escape_next:
                                    escape_next = False
                                elif char == '\\':
                                    escape_next = True
                                elif char == '"':
                                    in_string = False
                            else:
                                if char == '"':
                                    in_string = True
                                elif char == '{':
                                    depth -= 1
                                    if depth == 0:
                                        break
                                elif char == '}':
                                    depth += 1
                        
                        # 截断到最后一个完整对象的位置
                        pos = len(chunk_data) - len(buffer) + 1
                        chunk_data = chunk_data[:pos]
                        
                        # 回退文件指针，以确保下一个块从正确的位置开始
                        f.seek(-len(buffer) + 1, io.SEEK_CUR)
                    
                    # 确保每个块以有效的JSON格式开始（添加'['）和结束（添加']'）
                    # 但我们不希望处理字符串内的'[',']'
                    if chunk_index > 0:  # 非第一块
                        chunk_data = '[' + chunk_data
                    
                    # 所有块都加上']'以便ijson能正确解析
                    chunk_data = chunk_data + ']'
                    
                    # 提交处理任务到进程池
                    future = executor.submit(
                        process_pubmed_chunk, 
                        chunk_data, 
                        focal_ids, 
                        focal_entities, 
                        relation_schema, 
                        chunk_index
                    )
                    futures.append(future)
                    
                    chunk_index += 1
                    print(f"提交块 {chunk_index-1} 处理任务")
            
            # 收集所有结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="收集处理结果"):
                try:
                    relations, connections = future.result()
                    all_relations.extend(relations)
                    all_connections.append(connections)
                except Exception as e:
                    print(f"获取任务结果时出错: {e}")
        
        # 合并连接信息
        merged_connections = merge_entity_connections(all_connections)
        
        print(f"PubMed处理完成，找到 {len(all_relations)} 个关系")
    
    except Exception as e:
        print(f"处理PubMed文件时出错: {e}")
        print(tb.format_exc())
        return [], {}
    
    return all_relations, merged_connections

def simple_pubmed_processing_optimized(file_path, focal_ids, focal_entities, relation_schema):
    """
    优化的简单PubMed关系处理方法，使用ijson进行流处理
    """
    print("尝试使用优化的PubMed处理方法...")
    
    extracted_relations = []
    entity_connections = {
        'drug': {drug_id: set() for drug_id in focal_entities.get('drug', [])},
        'disease': {disease_id: set() for disease_id in focal_entities.get('disease', [])}
    }
    
    try:
        # 使用ijson以流方式处理JSON
        with open(file_path, 'rb') as f:
            # 使用ijson.items直接获取数组中的对象
            objects = ijson.items(f, 'item')
            
            # 使用进程池并行处理对象
            process_func = partial(
                process_relation_object, 
                focal_ids=focal_ids, 
                focal_entities=focal_entities, 
                relation_schema=relation_schema
            )
            
            object_count = 0
            relation_count = 0
            
            # 以批处理方式处理对象，每次处理BATCH_SIZE个对象
            BATCH_SIZE = 1000
            batch = []
            
            with tqdm(desc="处理PubMed关系") as pbar:
                for obj in objects:
                    batch.append(obj)
                    object_count += 1
                    
                    # 当达到批处理大小时，并行处理该批次
                    if len(batch) >= BATCH_SIZE:
                        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                            results = list(executor.map(process_func, batch))
                        
                        # 处理结果
                        for relations, connections in results:
                            extracted_relations.extend(relations)
                            relation_count += len(relations)
                            
                            # 更新连接
                            for drug_id in connections.get('drug', {}):
                                if drug_id in entity_connections['drug']:
                                    entity_connections['drug'][drug_id].update(connections['drug'][drug_id])
                            
                            for disease_id in connections.get('disease', {}):
                                if disease_id in entity_connections['disease']:
                                    entity_connections['disease'][disease_id].update(connections['disease'][disease_id])
                        
                        # 更新进度条
                        pbar.update(len(batch))
                        pbar.set_description(f"已处理: {object_count} 对象, 找到: {relation_count} 关系")
                        
                        # 清空批处理列表
                        batch = []
                        
                        # 强制垃圾回收
                        gc.collect()
                
                # 处理最后一批
                if batch:
                    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        results = list(executor.map(process_func, batch))
                    
                    # 处理结果
                    for relations, connections in results:
                        extracted_relations.extend(relations)
                        relation_count += len(relations)
                        
                        # 更新连接
                        for drug_id in connections.get('drug', {}):
                            if drug_id in entity_connections['drug']:
                                entity_connections['drug'][drug_id].update(connections['drug'][drug_id])
                        
                        for disease_id in connections.get('disease', {}):
                            if disease_id in entity_connections['disease']:
                                entity_connections['disease'][disease_id].update(connections['disease'][disease_id])
                    
                    # 更新进度条
                    pbar.update(len(batch))
                    pbar.set_description(f"已处理: {object_count} 对象, 找到: {relation_count} 关系")
            
            print(f"优化处理完成，处理了 {object_count} 个对象，找到 {relation_count} 个关系")
    
    except Exception as e:
        print(f"优化处理时出错: {e}")
        print(tb.format_exc())
    
    return extracted_relations, entity_connections

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
    parser.add_argument('--process_count', type=int, default=mp.cpu_count()-1, help='并行处理的进程数')
    parser.add_argument('--low_memory', action='store_true', help='启用低内存模式')
    
    args = parser.parse_args()
    
    # 设置全局常量
    global MAX_WORKERS
    global CHUNK_SIZE
    
    # 根据用户参数设置最大工作进程数
    if args.process_count > 0:
        MAX_WORKERS = args.process_count
    
    # 根据用户参数设置块大小
    if args.chunk_size > 0:
        CHUNK_SIZE = args.chunk_size
    
    # 添加调试输出
    print("解析后的参数:")
    print("数据目录:", args.data_dir)
    print("输出目录:", args.output_dir)
    print("药物关键词:", args.drug_keywords)
    print("疾病关键词:", args.disease_keywords)
    print("精确匹配:", args.exact_match)
    print("使用进程数:", MAX_WORKERS)
    print("块大小:", CHUNK_SIZE)
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
            # 使用通用关键词提取实体 - 如果有实体类型指定，先过滤一次
            if args.entity_types:
                filtered_nodes = [node for node in nodes_data if isinstance(node, dict) and node.get("type") in args.entity_types]
                print(f"按实体类型预过滤后的节点数: {len(filtered_nodes)}")
                nodes_to_process = filtered_nodes
            else:
                nodes_to_process = nodes_data
            
            all_entities, entity_id_map = extract_keyword_entities(
                nodes_to_process,
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
        
        # 释放内存 - 我们不再需要完整的节点数据
        del nodes_data
        gc.collect()
        
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
        
        # 提取PubMed关系 - 修改后的部分，使用新的并行和流处理方法
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
            
            # 尝试使用并行流处理方法
            pubmed_relations, entity_connections = process_pubmed_relations_parallel(
                pubmed_file,
                focal_ids,
                simple_focal_entities,
                relation_schema
            )
            
            # 如果没有找到关系，尝试优化的简单处理方法
            if not pubmed_relations:
                print("使用并行流处理未找到关系，尝试优化的简单处理方法...")
                pubmed_relations, entity_connections = simple_pubmed_processing_optimized(
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

        # 高效处理关系ID并保存
        if all_relations:
            # 创建并行映射函数
            def assign_entity_ids(focal_ids, all_relations):
                entity_mapping = {}
                next_id = 1
                
                # 首先为焦点实体分配ID
                for focal_id in focal_ids:
                    entity_mapping[focal_id] = next_id
                    next_id += 1
                
                # 收集所有需要分配ID的实体
                entity_ids = set()
                for relation in all_relations:
                    source_id = relation.get("Original Source ID")
                    target_id = relation.get("Original Target ID")
                    
                    if source_id and source_id not in entity_mapping:
                        entity_ids.add(source_id)
                    
                    if target_id and target_id not in entity_mapping:
                        entity_ids.add(target_id)
                
                # 为其他实体分配ID
                for entity_id in entity_ids:
                    entity_mapping[entity_id] = next_id
                    next_id += 1
                
                return entity_mapping
            
            # 分配实体ID
            print("使用高效ID映射...")
            entity_mapping = assign_entity_ids(focal_ids, all_relations)
            
            # 使用并行处理批量更新关系
            def update_relation_batch(relations_batch, entity_mapping):
                updated_batch = []
                for relation in relations_batch:
                    source_id = relation.get("Original Source ID")
                    target_id = relation.get("Original Target ID")
                    
                    if source_id in entity_mapping and target_id in entity_mapping:
                        relation_copy = relation.copy()
                        relation_copy["Source ID"] = entity_mapping[source_id]
                        relation_copy["Target ID"] = entity_mapping[target_id]
                        updated_batch.append(relation_copy)
                
                return updated_batch
            
            # 将关系分成多个批次
            batch_size = max(1, len(all_relations) // (MAX_WORKERS * 2))
            relation_batches = [all_relations[i:i+batch_size] for i in range(0, len(all_relations), batch_size)]
            
            # 并行处理每个批次
            updated_relations = []
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # 准备函数和参数
                func = partial(update_relation_batch, entity_mapping=entity_mapping)
                
                # 并行执行批处理
                for batch_result in tqdm(executor.map(func, relation_batches), total=len(relation_batches), desc="更新关系ID"):
                    updated_relations.extend(batch_result)
            
            print(f"自动映射了 {len(entity_mapping)} 个实体ID")
            print(f"更新了 {len(updated_relations)} 条关系的实体ID")
    
            # 保存关系
            save_relations_to_csv(updated_relations, args.output_dir)
            print(f"保存了 {len(updated_relations)} 条关系")
        
        # 提取关联实体 - 使用并行处理优化这一部分
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
        
        # 重新加载实体数据 - 只有在需要时才加载
        if related_ids:
            logger.info("重新加载实体数据以提取相关实体...")
            nodes_data = load_json_file(
                nodes_file,
                chunk_size=args.chunk_size,
                low_memory=args.low_memory,
                method='auto'
            )
            
            if not nodes_data:
                logger.error("重新加载实体数据失败")
                # 继续使用已有的焦点实体
                related_entities = []
                related_id_map = {}
            else:
                # 为了提高效率，先按ID过滤节点
                filtered_nodes = []
                related_ids_set = set(related_ids)  # 转换为集合以加快查找
                
                for node in tqdm(nodes_data, desc="预过滤相关实体"):
                    if isinstance(node, dict) and node.get("biokdeid") in related_ids_set:
                        if not args.entity_types or node.get("type") in args.entity_types:
                            filtered_nodes.append(node)
                
                print(f"预过滤后的相关实体节点数: {len(filtered_nodes)}")
                
                # 提取相关实体
                related_entities, related_id_map = extract_entities_by_ids(
                    filtered_nodes,
                    list(related_ids),
                    args.entity_types if args.entity_types else None
                )
                
                # 释放内存
                del nodes_data
                del filtered_nodes
                gc.collect()
            
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
        print(tb.format_exc())
        logger.error(f"数据提取过程中发生错误: {e}")
        logger.error(tb.format_exc())
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
        print(tb.format_exc())
        sys.exit(1)