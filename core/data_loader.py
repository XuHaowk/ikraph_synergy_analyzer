#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块
提供高效的数据加载功能，从batch_main_xin_4_7.py移植而来
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import mmap 
import multiprocessing
import gc
import psutil
import warnings
from typing import Union, List, Dict, Generator

# 尝试导入可选依赖
try:
    import orjson
except ImportError:
    orjson = None

try:
    import rapidjson
except ImportError:
    rapidjson = None

# 禁用不相关的警告
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

def load_json_file(file_path, chunk_size=None, low_memory=False, method='auto'):
    """Smart JSON loader using RapidJSON for maximum performance"""
    try:
        logger.info(f"Loading file: {file_path}")
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"File size: {file_size:.2f} MB")

        # 检查是否可以使用RapidJSON
        if rapidjson is not None:
            # Configure RapidJSON options
            parse_options = rapidjson.PM_TRAILING_COMMAS | rapidjson.PM_COMMENTS | rapidjson.PM_NONE
            mmap_block_size = 104857600  # 100MB
            # For smaller files, load directly
            if file_size < 500:  # Less than 500MB
                with open(file_path, 'rb') as f:
                    content = f.read()
                    data = rapidjson.loads(content, parse_mode=parse_options)
                    
                    # Ensure data is a list
                    if isinstance(data, dict):
                        data = list(data.values())
                    elif not isinstance(data, list):
                        data = [data]
                    
                    logger.info(f"Loaded {len(data)} entries with RapidJSON")
                    return data
            
            # For larger files, stream process
            else:
                results = []
                for batch in load_large_json_streaming_rapidjson(file_path, batch_size=chunk_size or 100000):
                    results.extend(batch)
                
                logger.info(f"Loaded {len(results)} entries via RapidJSON streaming")
                return results
                
        # 如果没有RapidJSON，尝试orjson
        elif orjson is not None and method in ['auto', 'orjson']:
            logger.info("Using orjson for parsing")
            with open(file_path, 'rb') as f:
                content = f.read()
                data = orjson.loads(content)
                
                # Ensure data is a list
                if isinstance(data, dict):
                    data = list(data.values())
                elif not isinstance(data, list):
                    data = [data]
                
                logger.info(f"Loaded {len(data)} entries with orjson")
                return data
                
        # 如果都没有，回退到标准json
        else:
            logger.info("Using standard json for parsing")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Ensure data is a list
                if isinstance(data, dict):
                    data = list(data.values())
                elif not isinstance(data, list):
                    data = [data]
                
                logger.info(f"Loaded {len(data)} entries with standard json")
                return data

    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def stream_load_json(file_path, chunk_size=1000000):
    """以流的方式加载大型JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 先读取开头确定是列表还是对象
            start_pos = f.tell()
            first_char = f.read(1).strip()
            f.seek(start_pos)  # 回到起始位置
            
            if first_char == '[':  # 处理JSON列表
                # 读取开头的 '['
                f.read(1)
                comma = ""
                buffer = ""
                chunk = []
                
                for line in f:
                    buffer += line
                    # 初步解析以查找完整的JSON对象
                    open_braces = buffer.count('{')
                    close_braces = buffer.count('}')
                    
                    if open_braces == close_braces and open_braces > 0:
                        # 清理尾部的逗号
                        if buffer.rstrip().endswith(','):
                            buffer = buffer.rstrip()[:-1]
                        
                        try:
                            item = json.loads(comma + buffer)
                            chunk.append(item)
                            
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
                                
                            comma = ","
                            buffer = ""
                        except json.JSONDecodeError:
                            # 不完整的对象，继续读取
                            pass
                
                # 返回最后一个块
                if chunk:
                    yield chunk
            
            elif first_char == '{':  # 处理JSON对象
                data = json.load(f)
                # 对于对象，我们按键值对进行分块
                keys = list(data.keys())
                for i in range(0, len(keys), chunk_size):
                    chunk = {k: data[k] for k in keys[i:i+chunk_size]}
                    yield chunk
            
            else:
                logger.error(f"不支持的JSON格式: {first_char}")
                yield {}
                
    except Exception as e:
        logger.error(f"流式加载JSON文件 {file_path} 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        yield {}

def load_large_json_streaming_rapidjson(file_path: str, batch_size: int = 100000, entity_ids=None):
    """
    Ultra-fast streaming processor for large JSON files using RapidJSON
    """
    if rapidjson is None:
        logger.error("RapidJSON not available, please install 'python-rapidjson' package")
        return
        
    logger.info(f"Using RapidJSON stream reader for file: {file_path}")
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")

    try:
        # Set up RapidJSON parsing parameters for maximum speed
        parse_options = rapidjson.PM_TRAILING_COMMAS | rapidjson.PM_COMMENTS | rapidjson.PM_NONE
        
        # Use a buffered reader approach with RapidJSON
        with open(file_path, 'rb') as f:
            # Check file format
            first_char = f.read(1)
            f.seek(0)  # Reset to beginning
            
            if first_char == b'[':  # JSON array format
                # Skip the opening bracket
                f.read(1)
                
                # Initialize processing variables
                buffer_size = 200 * 1024 * 1024  # 200MB buffer
                items_batch = []
                object_count = 0
                
                # Process file in chunks with RapidJSON's streaming parser
                current_object = bytearray()
                depth = 0
                in_object = False
                in_string = False
                escape_next = False
                
                while True:
                    chunk = f.read(buffer_size)
                    if not chunk:
                        break
                    
                    i = 0
                    while i < len(chunk):
                        byte = chunk[i]
                        i += 1
                        
                        # String handling with escape sequences
                        if escape_next:
                            escape_next = False
                            current_object.append(byte)
                            continue
                        
                        if in_string:
                            if byte == 92:  # backslash
                                escape_next = True
                            elif byte == 34:  # double quote
                                in_string = False
                            current_object.append(byte)
                            continue
                        
                        if byte == 34:  # double quote
                            in_string = True
                            current_object.append(byte)
                            continue
                            
                        # Handle nested objects
                        if byte == 123:  # opening brace {
                            if not in_object:
                                in_object = True
                                current_object = bytearray([byte])
                            else:
                                current_object.append(byte)
                            depth += 1
                        elif byte == 125:  # closing brace }
                            current_object.append(byte)
                            depth -= 1
                            
                            # Complete object found
                            if depth == 0 and in_object:
                                try:
                                    # Parse with RapidJSON for maximum speed
                                    item = rapidjson.loads(current_object, parse_mode=parse_options)
                                    
                                    # Filter by entity IDs if provided
                                    if entity_ids and isinstance(item, dict) and "id" in item:
                                        rel_id = item["id"]
                                        parts = rel_id.split(".")
                                        if len(parts) >= 2:
                                            source_id = parts[0]
                                            target_id = parts[1]
                                            if source_id in entity_ids or target_id in entity_ids:
                                                items_batch.append(item)
                                    else:
                                        items_batch.append(item)
                                    
                                    object_count += 1
                                    
                                    # Return batch when it reaches threshold
                                    if len(items_batch) >= batch_size:
                                        yield items_batch
                                        items_batch = []
                                    
                                except Exception as e:
                                    pass  # Skip malformed objects
                                
                                # Reset state
                                in_object = False
                                current_object = bytearray()
                        elif in_object:
                            current_object.append(byte)
                
                # Return any remaining items
                if items_batch:
                    yield items_batch
                
                logger.info(f"Processed {object_count} JSON objects with RapidJSON")
            else:
                logger.error(f"Unsupported JSON format - expected array starting with '['")
                yield []
                
    except Exception as e:
        logger.error(f"Error streaming JSON file with RapidJSON: {e}")
        import traceback
        logger.error(traceback.format_exc())
        yield []

def stream_json_array(file_path: str, batch_size: int = 10000) -> Generator[List[Dict], None, None]:
    """
    流式读取大型JSON文件的生成器函数
    
    参数:
    - file_path: JSON文件路径
    - batch_size: 每批次读取的数据量
    
    返回:
    - 生成器，每次生成一批JSON数据
    """
    try:
        # 打开文件并检查文件格式
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取文件第一个非空白字符
            first_char = f.read(1).strip()
            f.seek(0)  # 重置文件指针

            # 根据文件开头的字符判断文件类型
            if first_char == '[':
                # JSON数组格式
                data = json.load(f)
                
                # 将大数组分批处理
                for i in range(0, len(data), batch_size):
                    yield data[i:i+batch_size]
            
            elif first_char == '{':
                # JSON对象格式
                data = json.load(f)
                keys = list(data.keys())
                
                # 将对象按批次生成
                for i in range(0, len(keys), batch_size):
                    batch_keys = keys[i:i+batch_size]
                    yield {k: data[k] for k in batch_keys}
            
            else:
                # 无法识别的JSON格式
                logger.error(f"不支持的JSON文件格式: {first_char}")
                yield []

    except Exception as e:
        logger.error(f"流式读取 JSON 文件 {file_path} 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        yield []

def _load_with_memory_mapping(file_path: str) -> List[Dict]:
    """
    使用内存映射方式加载大文件
    
    优点：
    - 不需要一次性将整个文件加载到内存
    - 可以处理超大文件
    - 访问速度快
    
    Parameters:
    - file_path: JSON文件路径
    
    Returns:
    - 解析后的数据列表
    """
    try:
        # 使用流式读取方法，避免一次性加载整个文件
        def json_parse_stream(mm):
            decoder = json.JSONDecoder()
            pos = 0
            data = []
            while True:
                try:
                    # 尝试解析下一个 JSON 对象
                    result, pos = decoder.raw_decode(mm[pos:].decode('utf-8'))
                    data.append(result)
                except json.JSONDecodeError:
                    # 如果无法解析，可能是到达文件末尾
                    break
            return data

        # 打开文件并使用内存映射
        with open(file_path, 'rb') as f:
            # 使用内存映射
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # 使用流式解析方法
            data = json_parse_stream(mm)
            
            # 关闭内存映射
            mm.close()
            
            # 确保返回列表
            return data if isinstance(data, list) else [data]
    
    except Exception as e:
        logger.error(f"内存映射加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def parallel_json_load(
    file_path: str, 
    num_processes: int = None, 
    chunk_size: int = 10_000_000
) -> List[Dict]:
    """
    并行加载大型JSON文件
    
    Parameters:
    - file_path: JSON文件路径
    - num_processes: 并行进程数
    - chunk_size: 每个进程处理的块大小
    
    Returns:
    - 合并后的数据列表
    """
    if num_processes is None:
        num_processes = os.cpu_count()
    
    file_size = os.path.getsize(file_path)
    chunk_size = min(chunk_size, file_size // num_processes)
    
    def load_chunk(start: int, size: int) -> List[Dict]:
        try:
            with open(file_path, 'rb') as f:
                f.seek(start)
                chunk = f.read(size)
                # 尝试使用orjson，如果可用的话
                if orjson is not None:
                    return orjson.loads(chunk) if chunk else []
                else:
                    return json.loads(chunk) if chunk else []
        except Exception as e:
            logger.error(f"加载文件块失败: {e}")
            return []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [(i*chunk_size, chunk_size) for i in range(num_processes)]
        results = pool.starmap(load_chunk, args)
    
    # 安全地合并结果
    merged_data = [
        item for chunk in results 
        if isinstance(chunk, list) 
        for item in chunk
    ]
    
    return merged_data

def enable_large_dataset_processing():
    """配置系统处理大型数据集"""
    # 获取系统内存信息
    mem_info = psutil.virtual_memory()
    logger.info(f"系统内存: 总计={mem_info.total/(1024**3):.1f}GB, 可用={mem_info.available/(1024**3):.1f}GB")
    
    # 为pandas设置较大的显示选项
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    
    # 禁用pandas的SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    
    # 配置matplotlib以处理大型图表
    try:
        import matplotlib as mpl
        mpl.rcParams['agg.path.chunksize'] = 10000
    except ImportError:
        pass
    
    # 设置numpy以支持大型数组
    np.set_printoptions(threshold=10000)
    
    # 设置CPU亲和性以提高性能
    try:
        p = psutil.Process()
        if hasattr(p, 'cpu_affinity'):
            p.cpu_affinity(list(range(psutil.cpu_count())))
    except Exception as e:
        logger.warning(f"无法设置CPU亲和性: {e}")
    
    logger.info("系统已配置为处理大型数据集") 

def process_pubmed_chunk_rapidjson(start_pos, end_pos, file_path, focal_ids, focal_entities, relation_schema, buffer_size):
    """Process a chunk of the PubMed file with RapidJSON for maximum speed with enhanced error handling"""
    import rapidjson
    from core.relation_extractor import extract_single_pubmed_relation_rapidjson
    
    result_relations = []
    # Initialize entity connections dictionary
    chunk_entity_connections = {
        'drug': {drug_id: set() for drug_id in focal_entities.get('drug', [])},
        'disease': {disease_id: set() for disease_id in focal_entities.get('disease', [])}
    }
    
    try:
        with open(file_path, 'rb') as f:
            # Use a smaller buffer size to avoid memory issues
            working_buffer_size = min(buffer_size, 5*1024*1024)  # Limit to 5MB chunks
            
            # Skip to the start position with error handling
            try:
                f.seek(start_pos)
            except OSError:
                # If seeking directly fails, try incremental approach
                f.seek(0)
                remaining = start_pos
                chunk_size = 1024 * 1024  # 1MB chunks
                while remaining > 0:
                    to_skip = min(remaining, chunk_size)
                    f.read(to_skip)
                    remaining -= to_skip
            
            # Find the next object boundary if not at start
            if start_pos > 0:
                # Find the beginning of a valid JSON object by looking for '{'
                chars_to_check = 1000  # limit how far we search
                found_start = False
                search_buffer = b''
                
                for _ in range(chars_to_check):
                    char = f.read(1)
                    if not char:
                        break
                    search_buffer += char
                    if char == b'{':
                        f.seek(f.tell() - 1)  # Move back to include the '{'
                        found_start = True
                        break
                
                if not found_start:
                    # No valid start found, return empty results
                    return result_relations, chunk_entity_connections
            
            # Process until reaching end position or EOF
            current_pos = f.tell()
            object_count = 0
            
            # Initialize parsing state
            current_object = bytearray()
            depth = 0
            in_object = False
            in_string = False
            escape_next = False
            
            while current_pos < end_pos:
                try:
                    # Read a smaller chunk of data to avoid memory errors
                    read_size = min(working_buffer_size, end_pos - current_pos)
                    buffer = f.read(read_size)
                    if not buffer:
                        break
                    
                    # Process each byte in the buffer
                    i = 0
                    while i < len(buffer):
                        byte = buffer[i]
                        i += 1
                        
                        # Handle string escape sequences
                        if escape_next:
                            escape_next = False
                            current_object.append(byte)
                            continue
                        
                        # Handle string literals (ignore braces inside strings)
                        if in_string:
                            if byte == 92:  # backslash
                                escape_next = True
                            elif byte == 34:  # double quote
                                in_string = False
                            current_object.append(byte)
                            continue
                        
                        if byte == 34:  # double quote
                            in_string = True
                            current_object.append(byte)
                            continue
                        
                        # Handle object boundaries
                        if byte == 123:  # opening brace {
                            if not in_object:
                                in_object = True
                                current_object = bytearray([byte])
                            else:
                                current_object.append(byte)
                            depth += 1
                        elif byte == 125:  # closing brace }
                            depth -= 1
                            current_object.append(byte)
                            
                            # Complete object found
                            if depth == 0 and in_object:
                                try:
                                    # Parse with RapidJSON for maximum speed
                                    item = rapidjson.loads(current_object)
                                    
                                    # Check if this relation involves focal entities
                                    if isinstance(item, dict) and "id" in item:
                                        rel_id = item["id"]
                                        from core.relation_extractor import parse_relation_id
                                        rel_components = parse_relation_id(rel_id, relation_schema)
                                        
                                        if rel_components:
                                            source_id = rel_components["source_id"]
                                            target_id = rel_components["target_id"]
                                            
                                            # Check if related to focal entities
                                            if source_id in focal_ids or target_id in focal_ids:
                                                # Extract relationship details
                                                extracted_relations = extract_single_pubmed_relation_rapidjson(item, relation_schema)
                                                result_relations.extend(extracted_relations)
                                                
                                                # Track connected entity for each focal entity
                                                connected_entity_id = target_id if source_id in focal_ids else source_id
                                                focal_entity_id = source_id if source_id in focal_ids else target_id
                                                
                                                # Track for drug focal entities
                                                if focal_entity_id in focal_entities.get('drug', []):
                                                    chunk_entity_connections['drug'][focal_entity_id].add(connected_entity_id)
                                                
                                                # Track for disease focal entities
                                                if focal_entity_id in focal_entities.get('disease', []):
                                                    chunk_entity_connections['disease'][focal_entity_id].add(connected_entity_id)
                                    
                                    object_count += 1
                                    
                                    # Periodically free memory - more frequent than before
                                    if object_count % 25000 == 0:  # Was 100000, now more frequent
                                        gc.collect()
                                    
                                except Exception as e:
                                    # More graceful handling of parsing errors
                                    pass
                                
                                # Reset state
                                in_object = False
                                current_object = bytearray()
                        elif in_object:
                            current_object.append(byte)
                    
                    # Update current position
                    current_pos = f.tell()
                    
                    # Periodically free memory
                    if current_pos % (5 * 1024 * 1024) == 0:  # Every 5MB
                        gc.collect()
                        
                except MemoryError:
                    # If we hit a memory error, try to recover by clearing our buffers
                    current_object = bytearray()
                    in_object = False
                    depth = 0
                    gc.collect()
                    
                    # Try to skip ahead a bit and continue
                    try:
                        f.seek(min(f.tell() + 1024, end_pos))  # Skip ahead 1KB
                        current_pos = f.tell()
                    except:
                        # If seeking fails, just break out of the loop
                        break
    
    except Exception as e:
        # Capture and log any exceptions
        return result_relations, chunk_entity_connections
    
    return result_relations, chunk_entity_connections

def parallel_process_pubmed_relations(file_path, focal_ids, focal_entities, relation_schema, num_processes=8, chunk_size=100000, buffer_size=50*1024*1024):
    """Parallel process PubMed relation data with improved chunking and error handling for Chinese characters"""
    import multiprocessing as mp
    import os
    import gc
    
    # Create a multiprocessing context that works well for Windows
    mp_ctx = mp.get_context('spawn')
    
    print(f"Starting parallel PubMed processing: {num_processes} processes, buffer: {buffer_size/1024/1024:.1f}MB")
    
    # Calculate file size and create smaller chunks for better memory management
    file_size = os.path.getsize(file_path)
    
    # Use smaller chunks to prevent memory errors - 25MB per chunk
    chunk_size_bytes = 25 * 1024 * 1024  # 25MB chunks
    num_chunks = (file_size // chunk_size_bytes) + 1
    
    print(f"Processing PubMed file in {num_chunks} chunks")
    
    # Create chunk positions for processing
    chunk_positions = []
    for i in range(num_chunks):
        start_pos = i * chunk_size_bytes
        end_pos = min((i + 1) * chunk_size_bytes, file_size)
        
        # Use smaller buffer size to prevent memory errors
        actual_buffer_size = min(buffer_size, chunk_size_bytes // 4)
        
        chunk_positions.append((start_pos, end_pos, file_path, focal_ids, focal_entities, relation_schema, actual_buffer_size))
    
    # Process in smaller batches to manage memory better
    all_relations = []
    combined_connections = {
        'drug': {drug_id: set() for drug_id in focal_entities.get('drug', [])},
        'disease': {disease_id: set() for disease_id in focal_entities.get('disease', [])}
    }
    
    # Process only a few chunks at a time (batch_size)
    batch_size = 2  # Process just 2 chunks at a time to reduce memory usage
    actual_processes = min(4, num_processes)  # Limit processes to 4 for stability
    
    # Process chunks in batches
    for i in range(0, len(chunk_positions), batch_size):
        batch_chunks = chunk_positions[i:i+batch_size]
        
        try:
            with mp_ctx.Pool(processes=min(actual_processes, len(batch_chunks))) as pool:
                results = pool.starmap(process_pubmed_chunk_rapidjson, batch_chunks)
                
                # Process results from this batch
                for relations, connections in results:
                    if relations:  # Only process if we got valid relations
                        print(f"Chunk processed successfully: found {len(relations)} relations")
                        all_relations.extend(relations)
                        
                        # Merge entity connections
                        for entity_type in ['drug', 'disease']:
                            for entity_id, connected_ids in connections[entity_type].items():
                                combined_connections[entity_type][entity_id].update(connected_ids)
                
                # Close and join pool to prevent resource leaks
                pool.close()
                pool.join()
                
            # Force garbage collection between batches
            gc.collect()
            
        except Exception as e:
            import traceback
            print(f"Error processing batch {i//batch_size}: {e}")
            print(traceback.format_exc())
            # Continue with the next batch
    
    print(f"Extracted {len(all_relations)} relations from PubMed")
    
    return all_relations, combined_connections