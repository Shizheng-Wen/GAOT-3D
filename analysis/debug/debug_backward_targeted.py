#!/usr/bin/env python3
"""
有针对性的反向传播问题调试
1. 测试只使用encoder部分
2. 测试标准GNN模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append('../../')
from src.data.pyg_datasets import VTKMeshDataset
from src.trainer.utils.default_set import DatasetConfig, merge_config, ModelConfig
from torch_geometric.loader import DataLoader
from src.model import init_model
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Batch

class EncoderOnlyModel(nn.Module):
    """只使用GAOT3D的encoder部分的模型"""
    
    def __init__(self, input_size, output_size, magno_config):
        super().__init__()
        from src.model.layers.magno import GNOEncoder
        
        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = magno_config.lifting_channels
        self.coord_dim = magno_config.gno_coord_dim
        
        # 创建latent tokens (简化版本)
        D, H, W = 32, 32, 32  # 与原始模型相同
        self.num_latent_tokens = D * H * W
        
        x_min, y_min, z_min = -1, -1, -1
        x_max, y_max, z_max = 1, 1, 1
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, D),
            torch.linspace(y_min, y_max, H),
            torch.linspace(z_min, z_max, W),
            indexing="ij"
        )
        
        internal_latent_tokens = torch.stack(meshgrid, dim=-1).reshape(-1, self.coord_dim)
        self.register_buffer('latent_tokens', internal_latent_tokens)
        
        # 只使用encoder
        self.encoder = GNOEncoder(
            in_channels=input_size,
            out_channels=self.node_latent_size,
            gno_config=magno_config
        )
        
        # 添加一个简单的输出层来产生最终输出
        self.output_layer = nn.Linear(self.node_latent_size, output_size)
        
    def forward(self, batch: Batch):
        """前向传播：只使用encoder + 简单线性层"""
        device = batch.pos.device
        num_graphs = batch.num_graphs
        
        # 准备latent tokens
        latent_tokens_batched = self.latent_tokens.to(device).repeat(num_graphs, 1)
        batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(self.num_latent_tokens)
        
        # 只做编码
        rndata = self.encoder(
            batch=batch,
            latent_tokens_pos=latent_tokens_batched,
            latent_tokens_batch_idx=batch_idx_latent
        )  # 输出形状: [B, M, C_lifted]
        
        # 简单的全局池化和线性变换来产生输出
        # 将[B, M, C_lifted]变为[B*M, C_lifted]，然后平均池化回到[B, C_lifted]
        B, M, C = rndata.shape
        rndata_flat = rndata.view(B * M, C)
        
        # 为每个图创建batch索引
        batch_idx_for_pooling = torch.arange(B, device=device).repeat_interleave(M)
        
        # 全局平均池化
        pooled = global_mean_pool(rndata_flat, batch_idx_for_pooling)  # [B, C_lifted]
        
        # 线性变换到输出尺寸
        output = self.output_layer(pooled)  # [B, output_size]
        
        # 扩展到所有物理节点（为了与数据匹配）
        total_nodes = batch.num_nodes
        output_expanded = output[batch.batch]  # [total_nodes, output_size]
        
        return output_expanded

class StandardGNNModel(nn.Module):
    """标准的GNN模型，用于排除框架问题"""
    
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 创建GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_size, hidden_size))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_size, hidden_size))
            
        self.convs.append(GCNConv(hidden_size, output_size))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, batch: Batch):
        """标准GCN前向传播"""
        x, edge_index = batch.x, batch.encoder_edge_index_s0  # 使用编码器边
        
        # 通过GCN层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 不在最后一层应用激活和dropout
                x = F.relu(x)
                x = self.dropout(x)
        
        return x

class StandardGATModel(nn.Module):
    """标准的GAT模型，用于排除框架问题"""
    
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3, heads=4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.heads = heads
        
        # 创建GAT层
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_size, hidden_size, heads=heads, dropout=0.1))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_size * heads, hidden_size, heads=heads, dropout=0.1))
            
        # 最后一层输出单头
        self.convs.append(GATConv(hidden_size * heads, output_size, heads=1, dropout=0.1))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, batch: Batch):
        """标准GAT前向传播"""
        x, edge_index = batch.x, batch.encoder_edge_index_s0  # 使用编码器边
        
        # 通过GAT层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 不在最后一层应用激活和dropout
                x = F.elu(x)
                x = self.dropout(x)
        
        return x

def load_config():
    """加载配置"""
    config_path = "/cluster/work/math/shiwen/GAOT-3D/config/drivaerml/pressure_test.json"
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def test_encoder_only():
    """测试只使用encoder的模型"""
    print("\n" + "="*60)
    print("测试 1: 只使用Encoder部分")
    print("="*60)
    
    config = load_config()
    
    # 创建encoder-only模型
    model_config = merge_config(ModelConfig, config['model'])
    breakpoint()
    model = EncoderOnlyModel(
        input_size=3,
        output_size=1,
        magno_config=model_config.args
    )
    
    model = model.cuda()
    model.train()
    
    print(f"Encoder-only模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试有问题的样本
    test_files = ['boundary_102', 'boundary_10']  # 已知的问题样本
    
    results = {}
    
    for filename in test_files:
        print(f"\n{'='*40}")
        print(f"测试样本: {filename}")
        print(f"{'='*40}")
        
        try:
            # 加载数据
            success = load_and_test_sample(model, config, filename, "Encoder-only")
            results[filename] = success
            
        except Exception as e:
            print(f"  ❌ 样本 {filename} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            results[filename] = False
            
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    print(f"\nEncoder-only测试结果:")
    for filename, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {filename}: {status}")
    
    return all(results.values())

def test_standard_gnn():
    """测试标准GNN模型"""
    print("\n" + "="*60)
    print("测试 2: 标准GNN模型")
    print("="*60)
    
    config = load_config()
    
    # 测试不同的标准模型
    models_to_test = [
        ("GCN", StandardGNNModel(input_size=3, output_size=1, hidden_size=128, num_layers=3)),
        ("GAT", StandardGATModel(input_size=3, output_size=1, hidden_size=64, num_layers=3, heads=4))
    ]
    
    test_files = ['boundary_102', 'boundary_10']  # 已知的问题样本
    
    all_results = {}
    
    for model_name, model in models_to_test:
        print(f"\n{'='*50}")
        print(f"测试模型: {model_name}")
        print(f"{'='*50}")
        
        model = model.cuda()
        model.train()
        
        print(f"{model_name}模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        model_results = {}
        
        for filename in test_files:
            print(f"\n{'='*30}")
            print(f"  测试样本: {filename}")
            print(f"{'='*30}")
            
            try:
                success = load_and_test_sample(model, config, filename, model_name)
                model_results[filename] = success
                
            except Exception as e:
                print(f"    ❌ 样本 {filename} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                model_results[filename] = False
                
            # 清理GPU内存
            torch.cuda.empty_cache()
        
        all_results[model_name] = model_results
        
        print(f"\n{model_name}测试结果:")
        for filename, success in model_results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"    {filename}: {status}")
    
    return all_results

def load_and_test_sample(model, config, filename, model_name):
    """加载并测试单个样本"""
    dataset_config = merge_config(DatasetConfig, config['dataset'])
    
    base_path = config['dataset']['base_path']
    order_file = os.path.join(base_path, f"order_{config['dataset']['processed_folder']}.txt")
    
    # 找到目标文件的索引
    with open(order_file, 'r') as f:
        all_files = f.read().strip().split('\n')
    
    if filename not in all_files:
        print(f"    ❌ 文件 {filename} 不在数据集中")
        return False
    target_idx = all_files.index(filename)
    print(f"    文件索引: {target_idx}")
    # 创建只包含这个样本的数据集
    
    dataset = VTKMeshDataset(
        root=base_path,
        order_file=order_file,
        dataset_config=dataset_config,
        split='train'
    )
    
    # 获取目标样本
    if target_idx >= len(dataset):
        print(f"    ❌ 索引超出范围: {target_idx} >= {len(dataset)}")
        return False
        
    data = dataset[target_idx]
    dataloader = DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    batch = batch.cuda()
    
    # 详细分析样本特征
    print(f"    样本详细信息:")
    print(f"      节点数: {batch.num_nodes:,}")
    print(f"      编码器边数: {batch.encoder_edge_index_s0.shape[1]:,}")
    print(f"      解码器边数: {batch.decoder_edge_index_s0.shape[1]:,}")
    print(f"      特征形状: {batch.x.shape}")
    print(f"      位置形状: {batch.pos.shape}")
    
    # 检查边索引的有效性
    encoder_max_idx = batch.encoder_edge_index_s0.max().item()
    decoder_max_idx = batch.decoder_edge_index_s0.max().item()
    print(f"      编码器最大索引: {encoder_max_idx} (应该 < {batch.num_nodes})")
    print(f"      解码器最大索引: {decoder_max_idx} (应该 < {batch.num_nodes})")
    
    # 测试前向传播
    print(f"\n    测试前向传播...")
    model.zero_grad()
    
    with torch.cuda.device(0):
        output = model(batch)
        print(f"      ✅ 前向传播成功，输出形状: {output.shape}")
        
        # 计算简单损失
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        print(f"      损失值: {loss.item():.6f}")
        
        # 检查输出和损失的有效性
        print(f"      输出范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
        print(f"      输出是否包含NaN: {torch.isnan(output).any().item()}")
        print(f"      输出是否包含Inf: {torch.isinf(output).any().item()}")
        
        # 测试反向传播
        print(f"\n    测试反向传播...")
        try:
            loss.backward()
            print(f"      ✅ {model_name}模型反向传播成功！")
            
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
            
            print(f"      梯度统计: {len([g for _, g in grad_norms if g > 0])} 个参数有非零梯度")
            
            # 检查是否有异常的梯度值
            if grad_norms:
                max_grad_norm = max(grad_norms, key=lambda x: x[1])
                min_grad_norm = min(grad_norms, key=lambda x: x[1])
                print(f"      梯度范数范围: [{min_grad_norm[1]:.6f}, {max_grad_norm[1]:.6f}]")
                
                if max_grad_norm[1] > 1000:
                    print(f"      ⚠️  发现异常大的梯度: {max_grad_norm[0]} = {max_grad_norm[1]}")
            
            if grad_stats:
                print(f"      ⚠️  发现异常梯度:")
                for name, stats in grad_stats.items():
                    print(f"        {name}: NaN={stats['nan']}, Inf={stats['inf']}")
            
            return True
            
        except RuntimeError as e:
            print(f"      ❌ {model_name}模型反向传播失败: {e}")
            if 'CUDA' in str(e):
                print(f"      这是CUDA内存访问错误！")
                
                # 尝试获取更多调试信息
                print("      尝试使用CPU进行反向传播...")
                try:
                    # 将模型和数据移到CPU
                    model_cpu = model.cpu()
                    batch_cpu = batch.cpu()
                    output_cpu = model_cpu(batch_cpu)
                    target_cpu = target.cpu()
                    loss_cpu = torch.nn.functional.mse_loss(output_cpu, target_cpu)
                    loss_cpu.backward()
                    print(f"      ✅ CPU反向传播成功，说明问题在CUDA操作中")
                    
                    # 将模型移回GPU
                    model = model.cuda()
                    
                except Exception as cpu_e:
                    print(f"      ❌ CPU反向传播也失败: {cpu_e}")
            
            return False

def compare_results(encoder_success, gnn_results):
    """比较测试结果"""
    print("\n" + "="*60)
    print("结果分析")
    print("="*60)
    
    print(f"\n1. Encoder-only模型测试结果: {'成功' if encoder_success else '失败'}")
    
    print(f"\n2. 标准GNN模型测试结果:")
    for model_name, results in gnn_results.items():
        success_count = sum(results.values())
        total_count = len(results)
        print(f"   {model_name}: {success_count}/{total_count} 个样本成功")
        for filename, success in results.items():
            status = "✅" if success else "❌"
            print(f"     {filename}: {status}")
    
    print(f"\n分析结论:")
    
    # 如果encoder成功但原模型失败
    if encoder_success:
        print("✅ Encoder部分工作正常，问题可能在Processor(Transformer)或Decoder部分")
    else:
        print("❌ Encoder部分就有问题，需要进一步检查MAGNO编码器的实现")
    
    # 如果标准GNN也失败
    all_gnn_failed = all(not any(results.values()) for results in gnn_results.values())
    if all_gnn_failed:
        print("❌ 标准GNN模型也失败，可能是数据或PyTorch Geometric框架本身的问题")
        print("   建议检查:")
        print("   - 数据预处理是否正确")
        print("   - 边索引是否有效")
        print("   - GPU内存是否足够")
        print("   - CUDA版本兼容性")
    else:
        print("✅ 标准GNN模型至少部分成功，说明不是框架问题")
        print("   问题很可能在GAOT3D模型的特定实现中")

if __name__ == "__main__":
    # 设置CUDA调试
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print("开始有针对性的反向传播问题调试...")
    
    # 测试1: 只使用encoder
    #encoder_success = test_encoder_only()

    # 测试2: 标准GNN模型
    gnn_results = test_standard_gnn()
    
    # 比较和分析结果
    #compare_results(encoder_success, gnn_results) 