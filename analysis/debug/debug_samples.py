#!/usr/bin/env python3
"""
诊断脚本：检查数据集中不同样本的完整性和一致性
"""

import torch
import os
import sys
sys.path.append('../')

from src.data.pyg_datasets import VTKMeshDataset
from src.trainer.utils.default_set import DatasetConfig
from src.trainer.utils.default_set import merge_config

def check_sample_integrity(data, sample_idx):
    """检查单个样本的完整性"""
    print(f"\n=== 样本 {sample_idx} 检查 ===")
    
    # 基本信息
    print(f"文件名: {data.filename}")
    print(f"节点数量: {data.num_nodes}")
    print(f"潜在节点数量: {data.num_latent_nodes}")
    
    # 检查主要张量
    print(f"\n主要张量信息:")
    if hasattr(data, 'x') and data.x is not None:
        print(f"  - x (特征): {data.x.shape}, dtype={data.x.dtype}, device={data.x.device}")
        has_nan = torch.isnan(data.x).any().item()
        has_inf = torch.isinf(data.x).any().item()
        print(f"    NaN={has_nan}, Inf={has_inf}")
        if has_nan or has_inf:
            print(f"    ⚠️  警告: x 包含无效值!")
    
    if hasattr(data, 'pos') and data.pos is not None:
        print(f"  - pos (位置): {data.pos.shape}, dtype={data.pos.dtype}, device={data.pos.device}")
        has_nan = torch.isnan(data.pos).any().item()
        has_inf = torch.isinf(data.pos).any().item()
        print(f"    NaN={has_nan}, Inf={has_inf}")
        if has_nan or has_inf:
            print(f"    ⚠️  警告: pos 包含无效值!")
    
    # 检查编码器边索引
    print(f"\n编码器信息:")
    if hasattr(data, 'encoder_edge_index_s0') and data.encoder_edge_index_s0 is not None:
        edge_index = data.encoder_edge_index_s0
        print(f"  - encoder_edge_index_s0: {edge_index.shape}, dtype={edge_index.dtype}")
        print(f"    最小索引: {edge_index.min().item()}, 最大索引: {edge_index.max().item()}")
        
        # 检查索引越界 - encoder_edge_index[0] 应该 < num_latent_nodes, [1] 应该 < num_nodes
        max_latent_idx = edge_index[0].max().item()
        max_physical_idx = edge_index[1].max().item()
        print(f"    潜在节点索引范围: 0-{max_latent_idx} (应该 < {data.num_latent_nodes})")
        print(f"    物理节点索引范围: 0-{max_physical_idx} (应该 < {data.num_nodes})")
        
        if max_latent_idx >= data.num_latent_nodes:
            print(f"    ⚠️  警告: 编码器潜在节点索引越界!")
        if max_physical_idx >= data.num_nodes:
            print(f"    ⚠️  警告: 编码器物理节点索引越界!")
    
    if hasattr(data, 'encoder_query_counts_s0') and data.encoder_query_counts_s0 is not None:
        query_counts = data.encoder_query_counts_s0
        print(f"  - encoder_query_counts_s0: {query_counts.shape}, dtype={query_counts.dtype}")
        print(f"    值范围: {query_counts.min().item()} ~ {query_counts.max().item()}")
        print(f"    总和: {query_counts.sum().item()}")
    
    # 检查解码器边索引
    print(f"\n解码器信息:")
    if hasattr(data, 'decoder_edge_index_s0') and data.decoder_edge_index_s0 is not None:
        edge_index = data.decoder_edge_index_s0
        print(f"  - decoder_edge_index_s0: {edge_index.shape}, dtype={edge_index.dtype}")
        print(f"    最小索引: {edge_index.min().item()}, 最大索引: {edge_index.max().item()}")
        
        # 检查索引越界 - decoder_edge_index[0] 应该 < num_nodes, [1] 应该 < num_latent_nodes
        max_physical_idx = edge_index[0].max().item()
        max_latent_idx = edge_index[1].max().item()
        print(f"    物理节点索引范围: 0-{max_physical_idx} (应该 < {data.num_nodes})")
        print(f"    潜在节点索引范围: 0-{max_latent_idx} (应该 < {data.num_latent_nodes})")
        
        if max_physical_idx >= data.num_nodes:
            print(f"    ⚠️  警告: 解码器物理节点索引越界!")
        if max_latent_idx >= data.num_latent_nodes:
            print(f"    ⚠️  警告: 解码器潜在节点索引越界!")
    
    if hasattr(data, 'decoder_query_counts_s0') and data.decoder_query_counts_s0 is not None:
        query_counts = data.decoder_query_counts_s0
        print(f"  - decoder_query_counts_s0: {query_counts.shape}, dtype={query_counts.dtype}")
        print(f"    值范围: {query_counts.min().item()} ~ {query_counts.max().item()}")
        print(f"    总和: {query_counts.sum().item()}")
    
    # 检查数据完整性
    print(f"\n数据完整性检查:")
    print(f"  - encoder_edge_index_s0 边数: {data.encoder_edge_index_s0.shape[1] if hasattr(data, 'encoder_edge_index_s0') and data.encoder_edge_index_s0 is not None else 0}")
    print(f"  - decoder_edge_index_s0 边数: {data.decoder_edge_index_s0.shape[1] if hasattr(data, 'decoder_edge_index_s0') and data.decoder_edge_index_s0 is not None else 0}")
    
    # 检查 query_counts 与实际边数的一致性
    if (hasattr(data, 'encoder_query_counts_s0') and data.encoder_query_counts_s0 is not None and
        hasattr(data, 'encoder_edge_index_s0') and data.encoder_edge_index_s0 is not None):
        expected_edges = data.encoder_query_counts_s0.sum().item()
        actual_edges = data.encoder_edge_index_s0.shape[1]
        print(f"  - 编码器边数一致性: 期望={expected_edges}, 实际={actual_edges}, 一致={expected_edges==actual_edges}")
        if expected_edges != actual_edges:
            print(f"    ⚠️  警告: 编码器边数不一致!")
    
    if (hasattr(data, 'decoder_query_counts_s0') and data.decoder_query_counts_s0 is not None and
        hasattr(data, 'decoder_edge_index_s0') and data.decoder_edge_index_s0 is not None):
        expected_edges = data.decoder_query_counts_s0.sum().item()
        actual_edges = data.decoder_edge_index_s0.shape[1]
        print(f"  - 解码器边数一致性: 期望={expected_edges}, 实际={actual_edges}, 一致={expected_edges==actual_edges}")
        if expected_edges != actual_edges:
            print(f"    ⚠️  警告: 解码器边数不一致!")

def compare_samples(data1, data2):
    """比较两个样本的结构"""
    print(f"\n=== 样本比较 ===")
    
    # 比较基本属性
    print(f"文件名: 样本1='{data1.filename}', 样本2='{data2.filename}', 相同={data1.filename==data2.filename}")
    print(f"num_nodes: 样本1={data1.num_nodes}, 样本2={data2.num_nodes}, 相同={data1.num_nodes==data2.num_nodes}")
    print(f"num_latent_nodes: 样本1={data1.num_latent_nodes}, 样本2={data2.num_latent_nodes}, 相同={data1.num_latent_nodes==data2.num_latent_nodes}")
    
    # 比较张量形状
    tensor_attrs = ['x', 'pos', 'encoder_edge_index_s0', 'encoder_query_counts_s0', 
                   'decoder_edge_index_s0', 'decoder_query_counts_s0']
    
    print(f"\n张量形状比较:")
    for attr in tensor_attrs:
        if hasattr(data1, attr) and hasattr(data2, attr):
            t1 = getattr(data1, attr)
            t2 = getattr(data2, attr)
            if t1 is not None and t2 is not None:
                print(f"  {attr}: 样本1={t1.shape}, 样本2={t2.shape}, 相同={t1.shape==t2.shape}")
                if attr.endswith('_edge_index_s0'):
                    # 对于边索引，还比较边数
                    edges1 = t1.shape[1] if len(t1.shape) > 1 else 0
                    edges2 = t2.shape[1] if len(t2.shape) > 1 else 0
                    print(f"    边数: 样本1={edges1}, 样本2={edges2}, 相同={edges1==edges2}")
            elif t1 is None and t2 is None:
                print(f"  {attr}: 两个样本都为None")
            else:
                print(f"  {attr}: 样本1={'None' if t1 is None else t1.shape}, 样本2={'None' if t2 is None else t2.shape}, ⚠️不匹配!")
    
    # 比较数据类型
    print(f"\n数据类型比较:")
    for attr in ['x', 'pos', 'encoder_edge_index_s0', 'decoder_edge_index_s0']:
        if (hasattr(data1, attr) and hasattr(data2, attr) and 
            getattr(data1, attr) is not None and getattr(data2, attr) is not None):
            dtype1 = getattr(data1, attr).dtype
            dtype2 = getattr(data2, attr).dtype
            print(f"  {attr}: 样本1={dtype1}, 样本2={dtype2}, 相同={dtype1==dtype2}")
            if dtype1 != dtype2:
                print(f"    ⚠️  警告: {attr} 数据类型不一致!")

def main():
    # 配置路径
    config_path = "../config/drivaerml/pressure.json"
    
    # 读取配置
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_config = merge_config(DatasetConfig, config['dataset'])
    dataset_config.train_size = 4

    # 创建数据集
    base_path = config['dataset']['base_path']
    order_file = os.path.join(base_path, "order_use.txt")
    dataset = VTKMeshDataset(
        root=base_path,
        order_file=order_file,
        dataset_config=dataset_config,
        split='train'
    )
    print(f"数据集大小: {len(dataset)}")
    
    # 检查前几个样本
    num_samples_to_check = min(4, len(dataset))
    samples = []
    
    for i in range(num_samples_to_check):
        try:
            print(f"\n正在加载样本 {i}...")
            data = dataset[i]
            samples.append(data)
            check_sample_integrity(data, i)
        except Exception as e:
            print(f"❌ 样本 {i} 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 比较样本
    if len(samples) >= 2:
        compare_samples(samples[0], samples[1])

if __name__ == "__main__":
    main() 