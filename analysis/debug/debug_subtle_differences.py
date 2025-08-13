#!/usr/bin/env python3
"""
详细比较问题样本和正常样本之间的细微差异
"""

import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import gc
import os
import sys

# 添加项目路径
sys.path.append('/cluster/work/math/shiwen/GAOT-3D')

from src.data.pyg_datasets import VTKMeshDataset
from src.trainer.utils.default_set import DatasetConfig, merge_config, ModelConfig
from src.model.gaot_3d import GAOT3D

def check_data_integrity(data, sample_name):
    """检查数据完整性"""
    print(f"\n=== {sample_name} 数据完整性检查 ===")
    
    # 检查基本属性
    print(f"  数据类型: {type(data)}")
    print(f"  设备: {data.device}")
    print(f"  是否在GPU: {data.is_cuda}")
    
    # 检查所有张量属性
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            
            # 检查NaN和Inf
            if value.dtype in [torch.float16, torch.float32, torch.float64]:
                nan_count = torch.isnan(value).sum().item()
                inf_count = torch.isinf(value).sum().item()
                if nan_count > 0 or inf_count > 0:
                    print(f"    ⚠️  发现异常值: NaN={nan_count}, Inf={inf_count}")
            
            # 检查内存连续性
            if not value.is_contiguous():
                print(f"    ⚠️  张量不连续: {key}")
            
            # 检查内存对齐
            if value.element_size() * value.numel() % 8 != 0:
                print(f"    ⚠️  内存未对齐: {key}")

def check_edge_index_integrity(data, sample_name):
    """检查边索引的完整性"""
    print(f"\n=== {sample_name} 边索引检查 ===")
    
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    print(f"  边索引形状: {edge_index.shape}")
    print(f"  节点数: {num_nodes}")
    print(f"  边数: {edge_index.shape[1]}")
    
    # 检查索引范围
    min_idx = edge_index.min().item()
    max_idx = edge_index.max().item()
    print(f"  索引范围: [{min_idx}, {max_idx}]")
    
    if min_idx < 0 or max_idx >= num_nodes:
        print(f"  ❌ 索引越界!")
        return False
    
    # 检查重复边
    unique_edges = edge_index.t().unique(dim=0)
    if unique_edges.shape[0] != edge_index.shape[1]:
        print(f"  ⚠️  存在重复边: {edge_index.shape[1]} -> {unique_edges.shape[0]}")
    
    # 检查自环
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    if self_loops > 0:
        print(f"  ⚠️  存在自环: {self_loops}")
    
    return True

def check_memory_layout(data, sample_name):
    """检查内存布局"""
    print(f"\n=== {sample_name} 内存布局检查 ===")
    
    total_memory = 0
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            memory_size = value.element_size() * value.numel()
            total_memory += memory_size
            print(f"  {key}: {memory_size / 1024**2:.2f} MB")
    
    print(f"  总内存: {total_memory / 1024**2:.2f} MB")
    
    # 检查内存碎片
    if hasattr(torch.cuda, 'memory_stats'):
        stats = torch.cuda.memory_stats()
        print(f"  GPU内存碎片: {stats.get('allocated_bytes.all.fragmented', 0) / 1024**2:.2f} MB")

def compare_samples():
    """比较不同样本的差异"""
    print("=== 详细样本比较 ===")
    
    # 加载数据集
    config_path = "config/drivaerml/pressure.json"
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    dataset_config = merge_config(DatasetConfig, config['dataset'])
    dataset_config.train_size = 4
    base_path = config['dataset']['base_path']
    order_file = os.path.join(base_path, "order_use.txt")
    dataset = VTKMeshDataset(
        root=base_path,
        order_file=order_file,
        dataset_config=dataset_config,
        split='train'
    )
        
    # 测试样本列表
    test_samples = ['boundary_101', 'boundary_100', 'boundary_102', 'boundary_10']
    
    results = {}
    with open(order_file, 'r') as f:
                all_files = f.read().strip().split('\n')

    for sample_name in test_samples:
        print(f"\n{'='*60}")
        print(f"分析样本: {sample_name}")
        print(f"{'='*60}")
        
        try:
            # 加载样本
            target_idx = all_files.index(sample_name)
            sample_data = dataset[target_idx]
            breakpoint()
            # 基本信息
            print(f"样本信息:")
            print(f"  节点数: {sample_data.num_nodes:,}")
            print(f"  边数: {sample_data.edge_index.shape[1]:,}")
            print(f"  特征维度: {sample_data.x.shape[1] if hasattr(sample_data, 'x') else 'N/A'}")
            
            # 数据完整性检查
            check_data_integrity(sample_data, sample_name)
            
            # 边索引检查
            edge_valid = check_edge_index_integrity(sample_data, sample_name)
            
            # 内存布局检查
            check_memory_layout(sample_data, sample_name)
            
            # 尝试移动到GPU
            print(f"\n--- 尝试移动到GPU ---")
            try:
                gpu_data = sample_data.cuda()
                print(f"  ✅ GPU移动成功")
                
                # 检查GPU上的数据
                check_data_integrity(gpu_data, f"{sample_name} (GPU)")
                
                # 尝试简单的张量操作
                if hasattr(gpu_data, 'x'):
                    test_op = gpu_data.x.sum()
                    print(f"  ✅ 张量操作成功: {test_op.item():.6f}")
                
                results[sample_name] = {
                    'status': 'success',
                    'edge_valid': edge_valid,
                    'gpu_ready': True
                }
                
            except Exception as e:
                print(f"  ❌ GPU移动失败: {e}")
                results[sample_name] = {
                    'status': 'gpu_failed',
                    'edge_valid': edge_valid,
                    'gpu_ready': False,
                    'error': str(e)
                }
            
            # 清理内存
            del sample_data
            if 'gpu_data' in locals():
                del gpu_data
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ 样本加载失败: {e}")
            results[sample_name] = {
                'status': 'load_failed',
                'error': str(e)
            }
    
    # 总结比较
    print(f"\n{'='*60}")
    print(f"比较总结")
    print(f"{'='*60}")
    
    for sample_name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_icon} {sample_name}: {result['status']}")
        if 'error' in result:
            print(f"    错误: {result['error']}")

def check_cuda_memory_patterns():
    """检查CUDA内存访问模式"""
    print(f"\n=== CUDA内存访问模式检查 ===")
    
    # 创建一个简单的测试张量
    test_tensor = torch.randn(1000, 1000).cuda()
    
    # 检查内存对齐
    print(f"测试张量内存对齐: {test_tensor.element_size() * test_tensor.numel() % 8}")
    
    # 检查CUDA内存统计
    if hasattr(torch.cuda, 'memory_stats'):
        stats = torch.cuda.memory_stats()
        print(f"CUDA内存统计:")
        for key, value in stats.items():
            if 'allocated' in key or 'reserved' in key:
                print(f"  {key}: {value / 1024**2:.2f} MB")
    
    del test_tensor
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=== 详细样本差异分析 ===")
    
    # 检查CUDA内存模式
    check_cuda_memory_patterns()
    
    # 比较样本
    compare_samples()
    
    print(f"\n分析完成!") 