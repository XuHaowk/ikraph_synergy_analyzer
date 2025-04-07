 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全局配置参数模块
定义系统运行所需的全局配置参数
"""

import os
import multiprocessing

# 数据加载配置
DATA_LOADING = {
    'chunk_size': 1000000,
    'buffer_size': 50*1024*1024,
    'parallel_chunks': 24,
    'process_count': max(1, multiprocessing.cpu_count() * 3 // 4),  # 默认使用75%的CPU核心
    'low_memory': False,
    'load_method': 'auto'
}

# 分析配置
ANALYSIS = {
    'min_confidence': 0.5,
    'max_path_length': 2,
    'psr_algorithm': 'default',
    'viz_sample': 1000,
    'exact_match': False
}

# 路径配置
PATHS = {
    'data_dir': './data',
    'output_dir': './output',
    'tables_dir': './output/tables',
    'graphs_dir': './output/graphs',
    'reports_dir': './output/reports',
}

# 确保输出目录存在
for path in [PATHS['output_dir'], PATHS['tables_dir'], PATHS['graphs_dir'], PATHS['reports_dir']]:
    os.makedirs(path, exist_ok=True)

# 日志配置
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'extraction.log'
}

# 实体类型配置
ENTITY_TYPES = {
    'DRUG': ['Chemical', 'Drug'],
    'DISEASE': ['Disease'],
    'GENE': ['Gene', 'Protein'],
    'PATHWAY': ['Pathway'],
    'ALL': []  # 空列表表示所有类型
}

# 关系条件配置
RELATION_FILTERS = {
    'min_documents': 1,  # 最少文献支持数量
    'min_score': 0.0,    # 最小分数
    'recent_only': False, # 是否仅使用最近的关系
    'years_threshold': 5  # 如果recent_only为True，仅使用最近几年的数据
}