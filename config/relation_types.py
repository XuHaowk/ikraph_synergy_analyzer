 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
关系类型定义模块
基于RelTypeInt.json定义关系类型和方向性
"""

import os
import json
import logging

# 关系方向常量
DIRECTION_POSITIVE = 1   # 正向关系，如激活、促进
DIRECTION_NEGATIVE = -1  # 负向关系，如抑制、减轻
DIRECTION_NEUTRAL = 0    # 中性关系，如关联但无明确方向
DIRECTION_UNKNOWN = None # 未知方向

# 预定义的关系类型映射
RELATION_TYPES = {
    "1": {"name": "Association", "direction": DIRECTION_NEUTRAL, "precision": 0.5},
    "2": {"name": "Positive_Correlation", "direction": DIRECTION_POSITIVE, "precision": 0.8},
    "3": {"name": "Negative_Correlation", "direction": DIRECTION_NEGATIVE, "precision": 0.8},
    "5": {"name": "Cotreatment", "direction": DIRECTION_NEUTRAL, "precision": 0.7},
    "7": {"name": "Drug_Interaction", "direction": DIRECTION_NEUTRAL, "precision": 0.8},
    "10": {"name": "Disease_Gene", "direction": DIRECTION_NEUTRAL, "precision": 0.7},
    "11": {"name": "Causes", "direction": DIRECTION_POSITIVE, "precision": 0.8},
    "12": {"name": "Regulates", "direction": [DIRECTION_POSITIVE, DIRECTION_NEGATIVE], "precision": 0.7},
    "16": {"name": "Treats", "direction": DIRECTION_NEGATIVE, "precision": 0.9},
    "19": {"name": "Palliates", "direction": DIRECTION_NEGATIVE, "precision": 0.8},
    "20": {"name": "Chemical_Gene", "direction": [DIRECTION_POSITIVE, DIRECTION_NEGATIVE], "precision": 0.7},
    "21": {"name": "Drug_Protein", "direction": [DIRECTION_POSITIVE, DIRECTION_NEGATIVE], "precision": 0.7},
    "23": {"name": "Chemical_Disease", "direction": [DIRECTION_POSITIVE, DIRECTION_NEGATIVE], "precision": 0.7},
    "25": {"name": "Drug_Drug", "direction": DIRECTION_NEUTRAL, "precision": 0.7},
    "28": {"name": "Disease_Phenotype_Negative", "direction": DIRECTION_NEGATIVE, "precision": 0.7},
    "29": {"name": "Disease_Phenotype_Positive", "direction": DIRECTION_POSITIVE, "precision": 0.7},
    "32": {"name": "Drug_Effect", "direction": [DIRECTION_POSITIVE, DIRECTION_NEGATIVE], "precision": 0.7},
    "40": {"name": "Exposure_Disease", "direction": DIRECTION_POSITIVE, "precision": 0.7},
    "45": {"name": "Pathway_Pathway", "direction": DIRECTION_NEUTRAL, "precision": 0.7},
    "46": {"name": "Pathway_Protein", "direction": DIRECTION_NEUTRAL, "precision": 0.7},
    "48": {"name": "Anatomy_Protein_Present", "direction": DIRECTION_POSITIVE, "precision": 0.8},
    "49": {"name": "Anatomy_Protein_Absent", "direction": DIRECTION_NEGATIVE, "precision": 0.8},
    "50": {"name": "Drug_Target", "direction": [DIRECTION_POSITIVE, DIRECTION_NEGATIVE, DIRECTION_NEUTRAL], "precision": 0.8},
    "51": {"name": "Target_Disease", "direction": [DIRECTION_POSITIVE, DIRECTION_NEGATIVE], "precision": 0.7},
    "52": {"name": "Biomarker_Disease", "direction": [DIRECTION_POSITIVE, DIRECTION_NEGATIVE], "precision": 0.7}
}

# 重点关系类型分组
KEY_RELATION_TYPES = {
    "drug_target": ["21", "50"],  # 药物-靶点关系
    "drug_disease": ["16", "19", "23"],  # 药物-疾病关系
    "gene_disease": ["10", "51", "52"],  # 基因-疾病关系
    "toxicity": ["11", "23", "40"],  # 毒性相关关系
    "synergy": ["5", "7", "25"],  # 协同相关关系
    "regulation": ["2", "3", "12"]  # 调节关系
}

def load_relation_types(schema_file):
    """
    从RelTypeInt.json加载完整的关系类型定义
    
    Parameters:
    - schema_file: RelTypeInt.json文件路径
    
    Returns:
    - 完整的关系类型映射字典
    """
    relation_types = RELATION_TYPES.copy()  # 从预定义映射开始
    
    try:
        if not os.path.exists(schema_file):
            logging.warning(f"关系类型模式文件 {schema_file} 不存在，使用预定义映射")
            return relation_types
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # 更新关系类型映射
        for rel in schema_data:
            int_rep = rel["intRep"]
            rel_type = rel["relType"]
            cor_type = rel.get("corType", [0])  # 默认为中性
            
            # 确定方向
            if 1 in cor_type and -1 in cor_type:
                direction = [DIRECTION_POSITIVE, DIRECTION_NEGATIVE]
            elif 1 in cor_type:
                direction = DIRECTION_POSITIVE
            elif -1 in cor_type:
                direction = DIRECTION_NEGATIVE
            else:
                direction = DIRECTION_NEUTRAL
            
            # 更新或添加关系类型
            relation_types[int_rep] = {
                "name": rel_type,
                "direction": direction,
                "precision": rel.get("relPrec", 0.5)
            }
        
        logging.info(f"从 {schema_file} 加载了 {len(schema_data)} 种关系类型")
        return relation_types
    
    except Exception as e:
        logging.error(f"加载关系类型失败: {e}")
        return relation_types

def get_relation_direction(relation_type_id, context=None):
    """
    获取关系的方向
    
    Parameters:
    - relation_type_id: 关系类型ID
    - context: 关系上下文，用于判断方向（例如，对于治疗关系，正面上下文表示治疗效果）
    
    Returns:
    - 关系方向（1: 正向, -1: 负向, 0: 中性, None: 未知）
    """
    if relation_type_id not in RELATION_TYPES:
        return DIRECTION_UNKNOWN
    
    direction = RELATION_TYPES[relation_type_id]["direction"]
    
    # 如果方向是列表（多种可能），则根据上下文确定
    if isinstance(direction, list):
        # 根据上下文判断方向
        if context is not None:
            if relation_type_id == "23":  # Chemical_Disease
                # 根据上下文判断是治疗关系还是不良反应
                if context.get("is_treatment", False):
                    return DIRECTION_NEGATIVE  # 治疗是负向关系（减轻疾病）
                elif context.get("is_adverse", False):
                    return DIRECTION_POSITIVE  # 不良反应是正向关系（导致疾病）
            
            if relation_type_id in ["20", "21", "50"]:  # 药物-基因关系
                # 根据上下文判断是激活还是抑制
                if context.get("is_activation", False):
                    return DIRECTION_POSITIVE
                elif context.get("is_inhibition", False):
                    return DIRECTION_NEGATIVE
        
        # 如果无法确定，返回第一个方向作为默认值
        return direction[0]
    
    return direction

def is_positive_relation(relation_type_id, context=None):
    """
    判断是否为正向关系
    
    Parameters:
    - relation_type_id: 关系类型ID
    - context: 关系上下文
    
    Returns:
    - 是否为正向关系
    """
    return get_relation_direction(relation_type_id, context) == DIRECTION_POSITIVE

def is_negative_relation(relation_type_id, context=None):
    """
    判断是否为负向关系
    
    Parameters:
    - relation_type_id: 关系类型ID
    - context: 关系上下文
    
    Returns:
    - 是否为负向关系
    """
    return get_relation_direction(relation_type_id, context) == DIRECTION_NEGATIVE

def get_relation_precision(relation_type_id):
    """
    获取关系的精确度
    
    Parameters:
    - relation_type_id: 关系类型ID
    
    Returns:
    - 关系精确度（0-1的浮点数）
    """
    if relation_type_id not in RELATION_TYPES:
        return 0.5  # 默认中等精确度
    
    return RELATION_TYPES[relation_type_id].get("precision", 0.5)