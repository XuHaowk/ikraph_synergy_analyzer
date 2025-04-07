#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理工具模块
提供文本处理相关的工具函数
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

def clean_non_ascii_chars(text, preserve_greek=True):
    """
    高级文本清理，使用Unicode规范化处理希腊字母
    
    Parameters:
    - text: 要清理的文本
    - preserve_greek: 是否保留希腊字母
    
    Returns:
    - 清理后的文本
    """
    if not isinstance(text, str):
        return text
    
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

def normalize_text(text):
    """
    标准化文本，包括统一大小写、移除特殊字符等
    
    Parameters:
    - text: 要标准化的文本
    
    Returns:
    - 标准化后的文本
    """
    if not isinstance(text, str):
        return text
    
    # 转换为小写
    text = text.lower()
    
    # 替换特殊字符
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 替换多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_sentences(text):
    """
    将文本拆分为句子
    
    Parameters:
    - text: 要拆分的文本
    
    Returns:
    - 句子列表
    """
    if not isinstance(text, str):
        return []
    
    # 简单的句子分割规则
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def is_similar_string(s1, s2, threshold=0.8):
    """
    判断两个字符串是否相似
    
    Parameters:
    - s1: 第一个字符串
    - s2: 第二个字符串
    - threshold: 相似度阈值
    
    Returns:
    - 是否相似
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        return False
    
    # 标准化
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    
    # 如果字符串完全相同，直接返回True
    if s1 == s2:
        return True
    
    # 计算Jaccard相似度
    set1 = set(s1.split())
    set2 = set(s2.split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    
    return similarity >= threshold

def get_entity_context(text, entity_name, window_size=50):
    """
    获取实体在文本中的上下文
    
    Parameters:
    - text: 文本
    - entity_name: 实体名称
    - window_size: 上下文窗口大小
    
    Returns:
    - 上下文列表
    """
    if not isinstance(text, str) or not isinstance(entity_name, str):
        return []
    
    contexts = []
    
    # 查找实体在文本中的所有位置
    pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
    for match in pattern.finditer(text):
        start = max(0, match.start() - window_size)
        end = min(len(text), match.end() + window_size)
        context = text[start:end]
        contexts.append(context)
    
    return contexts 
