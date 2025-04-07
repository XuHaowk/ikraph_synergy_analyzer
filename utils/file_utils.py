#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件处理工具模块
提供文件操作相关的工具函数
"""

import os
import json
import csv
import logging
import pandas as pd
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

def ensure_directory(directory_path):
    """
    确保目录存在，如果不存在则创建
    
    Parameters:
    - directory_path: 目录路径
    
    Returns:
    - 目录路径
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"创建目录: {directory_path}")
    
    return directory_path

def check_directories(data_dir, output_dir):
    """检查并创建必要的目录"""
    # 检查数据目录
    if not os.path.exists(data_dir):
        logger.error(f"数据目录 {data_dir} 不存在！")
        return False
    
    # 确保数据目录存在
    if not os.listdir(data_dir):
        logger.error(f"数据目录 {data_dir} 为空！请添加iKraph数据文件。")
        return False
    
    # 检查必要的数据文件
    required_files = [
        "NER_ID_dict_cap_final.json", 
        "PubMedList.json", 
        "DBRelations.json", 
        "RelTypeInt.json"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            logger.error(f"缺少必要的数据文件: {file}")
            return False
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    # 创建子目录
    tables_dir = os.path.join(output_dir, "tables")
    graphs_dir = os.path.join(output_dir, "graphs")
    reports_dir = os.path.join(output_dir, "reports")
    
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
        logger.info(f"创建表格目录: {tables_dir}")
    
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
        logger.info(f"创建图形目录: {graphs_dir}")
    
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        logger.info(f"创建报告目录: {reports_dir}")
    
    return True

def save_to_csv(data, file_path, append=False):
    """
    保存数据到CSV文件
    
    Parameters:
    - data: 要保存的数据，可以是字典列表或DataFrame
    - file_path: 文件路径
    - append: 是否追加模式
    
    Returns:
    - 文件路径
    """
    # 确保目录存在
    ensure_directory(os.path.dirname(file_path))
    
    # 将数据转换为DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        logger.error(f"不支持的数据类型: {type(data)}")
        return file_path
    
    # 保存为CSV
    mode = 'a' if append else 'w'
    header = not append if os.path.exists(file_path) else True
    
    df.to_csv(file_path, mode=mode, header=header, index=False, encoding='utf-8')
    logger.info(f"{'追加' if append else '保存'} {len(df)} 条记录到 {file_path}")
    
    return file_path

def save_to_json(data, file_path, pretty=True):
    """
    保存数据到JSON文件
    
    Parameters:
    - data: 要保存的数据
    - file_path: 文件路径
    - pretty: 是否美化JSON格式
    
    Returns:
    - 文件路径
    """
    # 确保目录存在
    ensure_directory(os.path.dirname(file_path))
    
    # 保存为JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)
    
    logger.info(f"保存数据到 {file_path}")
    
    return file_path

def load_csv(file_path):
    """
    从CSV文件加载数据
    
    Parameters:
    - file_path: 文件路径
    
    Returns:
    - DataFrame
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"从 {file_path} 加载了 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"加载CSV文件 {file_path} 失败: {e}")
        return None

def load_json(file_path):
    """
    从JSON文件加载数据
    
    Parameters:
    - file_path: 文件路径
    
    Returns:
    - 加载的数据
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"从 {file_path} 加载数据")
        return data
    except Exception as e:
        logger.error(f"加载JSON文件 {file_path} 失败: {e}")
        return None

def save_entities_to_csv(entities, output_dir, file_name="entities.csv", append=False):
    """保存实体到CSV文件，支持追加模式"""
    if not entities:
        logger.warning("没有实体需要保存")
        return
    
    entities_file = os.path.join(output_dir, "tables", file_name)
    return save_to_csv(entities, entities_file, append)

def save_relations_to_csv(relations, output_dir, file_name="relations.csv", append=False):
    """保存关系到CSV文件，支持追加模式"""
    if not relations:
        logger.warning("没有关系需要保存")
        return
    
    relations_file = os.path.join(output_dir, "tables", file_name)
    return save_to_csv(relations, relations_file, append) 
