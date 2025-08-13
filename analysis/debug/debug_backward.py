#!/usr/bin/env python3
"""
调试反向传播问题
"""

import torch
import os
import sys
sys.path.append('src')

from src.data.pyg_datasets import VTKMeshDataset
from src.trainer.utils.default_set import DatasetConfig, merge_config, ModelConfig
from torch_geometric.loader import DataLoader
from src.model import init_model

def load_model_and_config():
    """加载模型和配置"""
    config_path = "config/drivaerml/pressure.json"
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 导入模型
    from src.model.gaot_3d import GAOT3D
    
    # 创建模型
    model_config = merge_config(ModelConfig, config['model'])
    model = init_model(
        input_size=3,
        output_size=1,
        model=model_config.name, 
        config=model_config.args
    )
    
    model = model.cuda()
    model.train()
    
    return model, config

def test_problematic_samples():
    """测试有问题的样本"""
    print("=== 测试有问题的样本 ===")
    
    model, config = load_model_and_config()
    
    # 设置环境变量以获得更详细的错误信息
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 要测试的样本
    test_files = ['boundary_101', 'boundary_100', 'boundary_10', 'boundary_102']
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    results = {}
    
    for filename in test_files:
        print(f"\n{'='*50}")
        print(f"测试样本: {filename}")
        print(f"{'='*50}")
        
        try:
            # 创建单样本数据集
            dataset_config = merge_config(DatasetConfig, config['dataset'])
            
            base_path = config['dataset']['base_path']
            order_file = os.path.join(base_path, "order_use.txt")
            
            # 找到目标文件的索引
            with open(order_file, 'r') as f:
                all_files = f.read().strip().split('\n')
            
            if filename not in all_files:
                print(f"  ❌ 文件 {filename} 不在数据集中")
                continue
                
            target_idx = all_files.index(filename)
            print(f"  文件索引: {target_idx}")
            
            # 创建只包含这个样本的数据集
            dataset_config.train_size = target_idx + 1  # 加载到目标索引
            
            dataset = VTKMeshDataset(
                root=base_path,
                order_file=order_file,
                dataset_config=dataset_config,
                split='train'
            )
            
            # 获取目标样本
            if target_idx >= len(dataset):
                print(f"  ❌ 索引超出范围: {target_idx} >= {len(dataset)}")
                continue
                
            data = dataset[target_idx]
            dataloader = DataLoader([data], batch_size=1, shuffle=False)
            batch = next(iter(dataloader))
            batch = batch.cuda()
            
            # 详细分析样本特征
            print(f"  样本详细信息:")
            print(f"    节点数: {batch.num_nodes:,}")
            print(f"    编码器边数: {batch.encoder_edge_index_s0.shape[1]:,}")
            print(f"    解码器边数: {batch.decoder_edge_index_s0.shape[1]:,}")
            print(f"    特征形状: {batch.x.shape}")
            print(f"    位置形状: {batch.pos.shape}")
            
            # 检查边索引的有效性
            encoder_max_idx = batch.encoder_edge_index_s0.max().item()
            decoder_max_idx = batch.decoder_edge_index_s0.max().item()
            print(f"    编码器最大索引: {encoder_max_idx} (应该 < {batch.num_nodes})")
            print(f"    解码器最大索引: {decoder_max_idx} (应该 < {batch.num_nodes})")
            
            # 检查数据范围
            print(f"    特征范围: [{batch.x.min().item():.6f}, {batch.x.max().item():.6f}]")
            print(f"    位置范围: [{batch.pos.min().item():.6f}, {batch.pos.max().item():.6f}]")
            
            # 测试前向传播
            print(f"\n  测试前向传播...")
            model.zero_grad()
            
            with torch.cuda.device(0):
                output = model(batch)
                print(f"    ✅ 前向传播成功，输出形状: {output.shape}")
                
                # 计算简单损失
                target = torch.randn_like(output)
                loss = torch.nn.functional.mse_loss(output, target)
                print(f"    损失值: {loss.item():.6f}")
                
                # 检查输出和损失的有效性
                print(f"    输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
                print(f"    输出是否包含NaN: {torch.isnan(output).any().item()}")
                print(f"    输出是否包含Inf: {torch.isinf(output).any().item()}")
                
                # 测试反向传播
                print(f"\n  测试反向传播...")
                try:
                    loss.backward()
                    print(f"    ✅ 反向传播成功")
                    
                    # 检查梯度
                    grad_norms = []
                    grad_stats = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_norms.append((name, grad_norm))
                            
                            # 检查梯度是否包含NaN或Inf
                            has_nan = torch.isnan(param.grad).any().item()
                            has_inf = torch.isinf(param.grad).any().item()
                            if has_nan or has_inf:
                                grad_stats[name] = {'nan': has_nan, 'inf': has_inf}
                    
                    print(f"    梯度统计: {len([g for _, g in grad_norms if g > 0])} 个参数有非零梯度")
                    
                    # 检查是否有异常的梯度值
                    if grad_norms:
                        max_grad_norm = max(grad_norms, key=lambda x: x[1])
                        min_grad_norm = min(grad_norms, key=lambda x: x[1])
                        print(f"    梯度范数范围: [{min_grad_norm[1]:.6f}, {max_grad_norm[1]:.6f}]")
                        
                        if max_grad_norm[1] > 1000:
                            print(f"    ⚠️  发现异常大的梯度: {max_grad_norm[0]} = {max_grad_norm[1]}")
                    
                    if grad_stats:
                        print(f"    ⚠️  发现异常梯度:")
                        for name, stats in grad_stats.items():
                            print(f"      {name}: NaN={stats['nan']}, Inf={stats['inf']}")
                    
                    results[filename] = {'status': 'success', 'grad_norms': grad_norms}
                    
                except RuntimeError as e:
                    print(f"    ❌ 反向传播失败: {e}")
                    if 'CUDA' in str(e):
                        print(f"    这是CUDA内存访问错误！")
                        
                        # 尝试获取更多调试信息
                        print("    尝试使用CPU进行反向传播...")
                        try:
                            # 将模型和数据移到CPU
                            model_cpu = model.cpu()
                            batch_cpu = batch.cpu()
                            output_cpu = model_cpu(batch_cpu)
                            target_cpu = target.cpu()
                            loss_cpu = torch.nn.functional.mse_loss(output_cpu, target_cpu)
                            loss_cpu.backward()
                            print(f"    ✅ CPU反向传播成功，说明问题在CUDA操作中")
                            
                            # 将模型移回GPU
                            model = model.cuda()
                            
                        except Exception as cpu_e:
                            print(f"    ❌ CPU反向传播也失败: {cpu_e}")
                    
                    results[filename] = {'status': 'failed', 'error': str(e)}
                    return False
                        
        except Exception as e:
            print(f"  ❌ 样本 {filename} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            results[filename] = {'status': 'error', 'error': str(e)}
            return False
            
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    # 总结结果
    print(f"\n{'='*60}")
    print(f"测试结果总结")
    print(f"{'='*60}")
    
    success_samples = []
    failed_samples = []
    
    for filename, result in results.items():
        if result['status'] == 'success':
            success_samples.append(filename)
            print(f"✅ {filename}: 成功")
        else:
            failed_samples.append(filename)
            print(f"❌ {filename}: 失败 - {result.get('error', 'Unknown error')}")
    
    print(f"\n成功样本: {success_samples}")
    print(f"失败样本: {failed_samples}")
    
    return len(failed_samples) == 0

def analyze_sample_differences():
    """分析正常样本和问题样本的差异"""
    print("\n=== 详细样本差异分析 ===")
    
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
    
    # 收集所有样本的统计信息
    sample_stats = {}
    
    for i in range(len(dataset)):
        data = dataset[i]
        filename = data.filename
        
        print(f"\n{'='*40}")
        print(f"样本 {i} ({filename}):")
        print(f"{'='*40}")
        
        # 基本统计
        stats = {
            'nodes': data.num_nodes,
            'encoder_edges': data.encoder_edge_index_s0.shape[1],
            'decoder_edges': data.decoder_edge_index_s0.shape[1],
            'filename': filename
        }
        
        print(f"  基本统计:")
        print(f"    节点数: {data.num_nodes:,}")
        print(f"    编码器边数: {data.encoder_edge_index_s0.shape[1]:,}")
        print(f"    解码器边数: {data.decoder_edge_index_s0.shape[1]:,}")
        
        # 特征统计
        if hasattr(data, 'x') and data.x is not None:
            x_mean = data.x.mean().item()
            x_std = data.x.std().item()
            x_min = data.x.min().item()
            x_max = data.x.max().item()
            
            stats.update({
                'x_mean': x_mean,
                'x_std': x_std,
                'x_min': x_min,
                'x_max': x_max
            })
            
            print(f"  特征统计:")
            print(f"    均值: {x_mean:.6f}")
            print(f"    标准差: {x_std:.6f}")
            print(f"    范围: [{x_min:.6f}, {x_max:.6f}]")
        
        # 位置统计
        if hasattr(data, 'pos') and data.pos is not None:
            pos_mean = data.pos.mean(dim=0)
            pos_min = data.pos.min(dim=0)[0]
            pos_max = data.pos.max(dim=0)[0]
            
            stats.update({
                'pos_mean_x': pos_mean[0].item(),
                'pos_mean_y': pos_mean[1].item(),
                'pos_mean_z': pos_mean[2].item(),
                'pos_min_x': pos_min[0].item(),
                'pos_min_y': pos_min[1].item(),
                'pos_min_z': pos_min[2].item(),
                'pos_max_x': pos_max[0].item(),
                'pos_max_y': pos_max[1].item(),
                'pos_max_z': pos_max[2].item()
            })
            
            print(f"  位置统计:")
            print(f"    均值: {pos_mean}")
            print(f"    范围: 最小={pos_min}, 最大={pos_max}")
        
        # 边索引统计
        encoder_degree = torch.bincount(data.encoder_edge_index_s0[1], minlength=data.num_nodes)
        decoder_degree = torch.bincount(data.decoder_edge_index_s0[1], minlength=data.num_nodes)
        
        encoder_degree_mean = encoder_degree.float().mean().item()
        encoder_degree_max = encoder_degree.max().item()
        decoder_degree_mean = decoder_degree.float().mean().item()
        decoder_degree_max = decoder_degree.max().item()
        
        stats.update({
            'encoder_degree_mean': encoder_degree_mean,
            'encoder_degree_max': encoder_degree_max,
            'decoder_degree_mean': decoder_degree_mean,
            'decoder_degree_max': decoder_degree_max
        })
        
        print(f"  度数统计:")
        print(f"    编码器度数: 均值={encoder_degree_mean:.2f}, 最大={encoder_degree_max}")
        print(f"    解码器度数: 均值={decoder_degree_mean:.2f}, 最大={decoder_degree_max}")
        
        # 内存估算
        approx_memory_mb = (
            data.x.numel() * 4 +  # 特征 (float32)
            data.pos.numel() * 4 +  # 位置 (float32) 
            data.encoder_edge_index_s0.numel() * 4 +  # 编码器边 (int32)
            data.decoder_edge_index_s0.numel() * 4   # 解码器边 (int32)
        ) / (1024 * 1024)
        
        stats['memory_mb'] = approx_memory_mb
        print(f"  估算内存使用: {approx_memory_mb:.1f} MB")
        
        # 检查边索引的有效性
        encoder_max_idx = data.encoder_edge_index_s0.max().item()
        decoder_max_idx = data.decoder_edge_index_s0.max().item()
        
        stats.update({
            'encoder_max_idx': encoder_max_idx,
            'decoder_max_idx': decoder_max_idx,
            'encoder_idx_valid': encoder_max_idx < data.num_nodes,
            'decoder_idx_valid': decoder_max_idx < data.num_nodes
        })
        
        print(f"  索引有效性:")
        print(f"    编码器最大索引: {encoder_max_idx} (有效: {encoder_max_idx < data.num_nodes})")
        print(f"    解码器最大索引: {decoder_max_idx} (有效: {decoder_max_idx < data.num_nodes})")
        
        sample_stats[filename] = stats
    
    # 比较分析
    print(f"\n{'='*60}")
    print(f"样本比较分析")
    print(f"{'='*60}")
    
    # 找出成功和失败的样本
    success_samples = ['boundary_101', 'boundary_100']
    failed_samples = ['boundary_102', 'boundary_10']
    
    print(f"成功样本: {success_samples}")
    print(f"失败样本: {failed_samples}")
    
    # 比较关键指标
    metrics = ['nodes', 'encoder_edges', 'decoder_edges', 'memory_mb', 
               'encoder_degree_mean', 'encoder_degree_max', 'decoder_degree_mean', 'decoder_degree_max']
    
    for metric in metrics:
        print(f"\n{metric} 比较:")
        success_values = [sample_stats[s][metric] for s in success_samples if metric in sample_stats[s]]
        failed_values = [sample_stats[s][metric] for s in failed_samples if metric in sample_stats[s]]
        
        if success_values and failed_values:
            success_avg = sum(success_values) / len(success_values)
            failed_avg = sum(failed_values) / len(failed_values)
            
            print(f"  成功样本平均值: {success_avg:.2f}")
            print(f"  失败样本平均值: {failed_avg:.2f}")
            print(f"  差异: {((failed_avg - success_avg) / success_avg * 100):+.1f}%")
            
            # 检查是否有明显模式
            if failed_avg > success_avg * 1.5:
                print(f"  ⚠️  失败样本的{metric}明显更大!")
    
    # 检查是否有异常值
    print(f"\n异常值检查:")
    for filename, stats in sample_stats.items():
        if filename in failed_samples:
            print(f"  {filename}:")
            for metric in ['encoder_degree_max', 'decoder_degree_max', 'memory_mb']:
                if metric in stats:
                    value = stats[metric]
                    if metric == 'memory_mb' and value > 350:
                        print(f"    ⚠️  {metric}: {value:.1f} MB (可能过大)")
                    elif 'degree_max' in metric and value > 10:
                        print(f"    ⚠️  {metric}: {value} (度数异常高)")

if __name__ == "__main__":
    # 设置CUDA调试
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print("开始调试反向传播问题...")
    test_problematic_samples()
    analyze_sample_differences() 