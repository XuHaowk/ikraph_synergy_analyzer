#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iKraph药物协同分析系统主程序
集成了数据提取、协同分析和毒性减轻分析功能
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ikraph_synergy.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="iKraph药物协同分析系统")
    
    # 操作模式
    parser.add_argument('--mode', type=str, choices=['extract', 'synergy', 'toxicity', 'all'], required=True,
                      help='运行模式: extract(数据提取), synergy(协同分析), toxicity(毒性减轻分析), all(全部)')
    
    # 目录参数
    parser.add_argument('--data_dir', type=str, default='./data', help='iKraph数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录路径')
    
    # 提取参数
    parser.add_argument('--drug1', type=str, help="第一种药物名称")
    parser.add_argument('--drug2', type=str, help="第二种药物名称")
    parser.add_argument('--disease', type=str, help="疾病名称")
    parser.add_argument('--exact_match', action='store_true', help='使用精确匹配（默认为部分匹配）')
    
    # 性能参数
    parser.add_argument('--chunk_size', type=int, default=1000000, help='大文件处理的块大小')
    parser.add_argument('--low_memory', action='store_true', help='启用低内存模式')
    
    return parser.parse_args()

def run_extract(args):
    """运行数据提取模式"""
    logger.info("运行数据提取模式...")
    
    # 构建extract_data.py的参数
    extract_args = [
        'python', 'scripts/extract_data.py',
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--chunk_size', str(args.chunk_size)
    ]
    
    # 添加药物和疾病关键词
    if args.drug1:
        extract_args.extend(['--drug_keywords', args.drug1])
    if args.drug2:
        extract_args.extend(['--drug_keywords', args.drug2])
    if args.disease:
        extract_args.extend(['--disease_keywords', args.disease])
    
    # 添加其他参数
    if args.exact_match:
        extract_args.append('--exact_match')
    if args.low_memory:
        extract_args.append('--low_memory')
    
    # 执行命令
    cmd = ' '.join(extract_args)
    logger.info(f"执行命令: {cmd}")
    return os.system(cmd) == 0

def run_synergy(args):
    """运行协同分析模式"""
    logger.info("运行协同分析模式...")
    
    # 检查必要参数
    if not args.drug1 or not args.drug2 or not args.disease:
        logger.error("协同分析模式需要指定drug1, drug2和disease参数")
        return False
    
    # 构建analyze_synergy.py的参数
    synergy_args = [
        'python', 'scripts/analyze_synergy.py',
        '--output_dir', args.output_dir,
        '--drug1', args.drug1,
        '--drug2', args.drug2,
        '--disease', args.disease
    ]
    
    # 添加其他参数
    if args.exact_match:
        synergy_args.append('--exact_match')
    
    # 执行命令
    cmd = ' '.join(synergy_args)
    logger.info(f"执行命令: {cmd}")
    return os.system(cmd) == 0

def run_toxicity(args):
    """运行毒性减轻分析模式"""
    logger.info("运行毒性减轻分析模式...")
    
    # 检查必要参数
    if not args.drug1 or not args.drug2:
        logger.error("毒性减轻分析模式需要指定drug1和drug2参数")
        return False
    
    # 构建analyze_toxicity.py的参数
    toxicity_args = [
        'python', 'scripts/analyze_toxicity.py',
        '--output_dir', args.output_dir,
        '--toxic_drug', args.drug1,
        '--protective_drug', args.drug2
    ]
    
    # 添加其他参数
    if args.exact_match:
        toxicity_args.append('--exact_match')
    
    # 执行命令
    cmd = ' '.join(toxicity_args)
    logger.info(f"执行命令: {cmd}")
    return os.system(cmd) == 0

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"开始处理时间: {start_time}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据模式运行相应功能
    if args.mode == 'extract' or args.mode == 'all':
        success = run_extract(args)
        if not success:
            logger.error("数据提取失败")
            return 1
    
    if args.mode == 'synergy' or args.mode == 'all':
        success = run_synergy(args)
        if not success:
            logger.error("协同分析失败")
            return 1
    
    if args.mode == 'toxicity' or args.mode == 'all':
        success = run_toxicity(args)
        if not success:
            logger.error("毒性减轻分析失败")
            return 1
    
    # 记录结束时间
    end_time = datetime.now()
    processing_time = end_time - start_time
    logger.info(f"结束处理时间: {end_time}")
    logger.info(f"总处理时间: {processing_time}")
    
    logger.info("处理完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 

