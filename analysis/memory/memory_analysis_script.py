#!/usr/bin/env python3
"""
显存分析脚本：探究MAGNO/GAOT-3D中edges数量对显存占用峰值的影响

主要分析：
1. 不同edges数量下的显存使用
2. 训练vs推理模式的显存差异  
3. encoder vs decoder的显存开销
4. 提供详细的分析报告和可视化

使用方法：
python memory_analysis_script.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Optional
import gc
import psutil
import time
from dataclasses import dataclass
from torch_geometric.data import Data, Batch
import warnings
warnings.filterwarnings('ignore')

# 添加src路径以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.layers.magno import MAGNOEncoder, MAGNODecoder, MAGNOConfig
from src.model.gaot_3d import GAOT3D
from src.model.layers.integral_transform import IntegralTransform


@dataclass
class MemoryTestConfig:
    """内存测试配置"""
    device: str = "cuda:0"
    batch_size: int = 1
    num_physical_nodes: int = 5000      # 物理网格节点数
    num_latent_tokens: int = 4096       # 潜在token数 (16x16x16)
    coord_dim: int = 3                  # 3D坐标
    input_channels: int = 4             # 输入特征维度
    output_channels: int = 1            # 输出特征维度
    lifting_channels: int = 64          # MAGNO通道数
    
    # edges数量测试范围
    min_edges: int = 10000             # 最小edges数
    max_edges: int = 1000000           # 最大edges数  
    num_test_points: int = 15          # 测试点数量
    
    # 测试选项
    test_encoder: bool = True
    test_decoder: bool = True
    test_full_model: bool = True
    test_training_mode: bool = True
    test_inference_mode: bool = True


class MemoryProfiler:
    """GPU显存监控器"""
    
    def __init__(self, device: str):
        self.device = device
        if torch.cuda.is_available() and device.startswith('cuda'):
            self.device_id = int(device.split(':')[1]) if ':' in device else 0
        else:
            raise ValueError("CUDA not available or invalid device")
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取当前显存信息 (MB)"""
        torch.cuda.synchronize(self.device_id)
        allocated = torch.cuda.memory_allocated(self.device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device_id) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(self.device_id) / 1024**2
        
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'max_allocated': max_allocated,
            'max_reserved': max_reserved
        }
    
    def reset_peak_memory(self):
        """重置峰值显存统计"""
        torch.cuda.reset_peak_memory_stats(self.device_id)
    
    def clear_cache(self):
        """清理显存缓存"""
        torch.cuda.empty_cache()
        gc.collect()


class SyntheticDataGenerator:
    """合成测试数据生成器"""
    
    @staticmethod
    def generate_batch(config: MemoryTestConfig, num_edges: int) -> Batch:
        """生成指定edges数量的PyG Batch"""
        
        # 生成物理节点位置
        phys_pos = torch.randn(config.num_physical_nodes, config.coord_dim, dtype=torch.float32)
        phys_feat = torch.randn(config.num_physical_nodes, config.input_channels, dtype=torch.float32)
        
        # 生成随机edges (确保不超过可能的最大edges数)
        max_possible_edges = config.num_physical_nodes * config.num_latent_tokens
        actual_num_edges = min(num_edges, max_possible_edges)
        
        # 随机生成edge_index
        edge_index = torch.randint(0, config.num_physical_nodes, (2, actual_num_edges), dtype=torch.long)
        
        # 创建batch
        data = Data(
            pos=phys_pos,
            x=phys_feat,
            edge_index=edge_index,
            batch=torch.zeros(config.num_physical_nodes, dtype=torch.long)
        )
        
        batch = Batch.from_data_list([data])
        return batch, actual_num_edges
    
    @staticmethod 
    def generate_latent_tokens(config: MemoryTestConfig, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成潜在tokens"""
        # 创建规律化的3D网格
        D = H = W = int(config.num_latent_tokens**(1/3))  # 假设立方体网格
        x = torch.linspace(-1, 1, D)
        y = torch.linspace(-1, 1, H) 
        z = torch.linspace(-1, 1, W)
        
        mesh_x, mesh_y, mesh_z = torch.meshgrid(x, y, z, indexing='ij')
        latent_pos = torch.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], dim=1)
        latent_batch = torch.zeros(latent_pos.shape[0], dtype=torch.long)
        
        return latent_pos.to(device), latent_batch.to(device)


class ComponentTester:
    """组件级显存测试器"""
    
    def __init__(self, config: MemoryTestConfig):
        self.config = config
        self.profiler = MemoryProfiler(config.device)
        self.device = config.device
        
        # 创建MAGNO配置
        self.magno_config = MAGNOConfig(
            gno_coord_dim=config.coord_dim,
            lifting_channels=config.lifting_channels,
            in_gno_channel_mlp_hidden_layers=[64, 64, 64],
            out_gno_channel_mlp_hidden_layers=[64, 64],
            precompute_edges=False  # 不使用预计算edges以便自由控制
        )
    
    def test_encoder_memory(self, num_edges: int, training_mode: bool = True) -> Dict[str, float]:
        """测试encoder显存使用"""
        self.profiler.clear_cache()
        self.profiler.reset_peak_memory()
        
        try:
            # 创建encoder
            encoder = MAGNOEncoder(
                in_channels=self.config.input_channels,
                out_channels=self.config.lifting_channels,
                gno_config=self.magno_config
            ).to(self.device)
            
            if training_mode:
                encoder.train()
            else:
                encoder.eval()
            
            # 生成测试数据
            batch, actual_edges = SyntheticDataGenerator.generate_batch(self.config, num_edges)
            batch = batch.to(self.device)
            
            latent_pos, latent_batch = SyntheticDataGenerator.generate_latent_tokens(self.config, self.device)
            
            initial_memory = self.profiler.get_memory_info()
            
            # 前向传播
            with torch.set_grad_enabled(training_mode):
                if training_mode:
                    # 训练模式：计算梯度
                    encoded = encoder(batch, latent_pos, latent_batch)
                    loss = encoded.sum()  # 简单的损失函数
                    loss.backward()
                else:
                    # 推理模式
                    with torch.no_grad():
                        encoded = encoder(batch, latent_pos, latent_batch)
            
            peak_memory = self.profiler.get_memory_info()
            
            # 清理
            del encoder, batch, encoded
            if training_mode and 'loss' in locals():
                del loss
            
            return {
                'actual_edges': actual_edges,
                'peak_allocated': peak_memory['max_allocated'],
                'peak_reserved': peak_memory['max_reserved'],
                'memory_increase': peak_memory['max_allocated'] - initial_memory['allocated']
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return {
                    'actual_edges': num_edges,
                    'peak_allocated': float('inf'),
                    'peak_reserved': float('inf'), 
                    'memory_increase': float('inf'),
                    'oom_error': True
                }
            else:
                raise e
    
    def test_decoder_memory(self, num_edges: int, training_mode: bool = True) -> Dict[str, float]:
        """测试decoder显存使用"""
        self.profiler.clear_cache()
        self.profiler.reset_peak_memory()
        
        try:
            # 创建decoder
            decoder = GNODecoder(
                in_channels=self.config.lifting_channels,
                out_channels=self.config.output_channels,
                gno_config=self.magno_config
            ).to(self.device)
            
            if training_mode:
                decoder.train()
            else:
                decoder.eval()
            
            # 生成测试数据
            batch, actual_edges = SyntheticDataGenerator.generate_batch(self.config, num_edges)
            batch = batch.to(self.device)
            
            latent_pos, latent_batch = SyntheticDataGenerator.generate_latent_tokens(self.config, self.device)
            
            # 创建潜在特征数据
            latent_features = torch.randn(
                self.config.num_latent_tokens, 
                self.config.lifting_channels,
                device=self.device
            )
            
            initial_memory = self.profiler.get_memory_info()
            
            # 前向传播
            with torch.set_grad_enabled(training_mode):
                if training_mode:
                    decoded = decoder(
                        rndata_flat=latent_features,
                        phys_pos_query=batch.pos,
                        batch_idx_phys_query=batch.batch,
                        latent_tokens_pos=latent_pos,
                        latent_tokens_batch_idx=latent_batch,
                        batch=None  # 不使用预计算edges
                    )
                    loss = decoded.sum()
                    loss.backward()
                else:
                    with torch.no_grad():
                        decoded = decoder(
                            rndata_flat=latent_features,
                            phys_pos_query=batch.pos,
                            batch_idx_phys_query=batch.batch,
                            latent_tokens_pos=latent_pos,
                            latent_tokens_batch_idx=latent_batch,
                            batch=None
                        )
            
            peak_memory = self.profiler.get_memory_info()
            
            # 清理
            del decoder, batch, decoded, latent_features
            if training_mode and 'loss' in locals():
                del loss
            
            return {
                'actual_edges': actual_edges,
                'peak_allocated': peak_memory['max_allocated'],
                'peak_reserved': peak_memory['max_reserved'],
                'memory_increase': peak_memory['max_allocated'] - initial_memory['allocated']
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return {
                    'actual_edges': num_edges,
                    'peak_allocated': float('inf'),
                    'peak_reserved': float('inf'),
                    'memory_increase': float('inf'),
                    'oom_error': True
                }
            else:
                raise e
    
    def test_full_model_memory(self, num_edges: int, training_mode: bool = True) -> Dict[str, float]:
        """测试完整GAOT-3D模型显存使用"""
        self.profiler.clear_cache()
        self.profiler.reset_peak_memory()
        
        try:
            # 创建完整模型 (简化的transformer配置)
            from src.model.layers.attn import TransformerConfig
            
            transformer_config = TransformerConfig(
                patch_size=4,  # 小patch避免显存过大
                hidden_size=256,
                num_layers=2,
                positional_embedding='absolute'
            )
            
            model = GAOT3D(
                input_size=self.config.input_channels,
                output_size=self.config.output_channels,
                magno_config=self.magno_config,
                attn_config=transformer_config,
                latent_tokens=(16, 16, 16)  # 4096 tokens
            ).to(self.device)
            
            if training_mode:
                model.train()
            else:
                model.eval()
            
            # 生成测试数据
            batch, actual_edges = SyntheticDataGenerator.generate_batch(self.config, num_edges)
            batch = batch.to(self.device)
            
            initial_memory = self.profiler.get_memory_info()
            
            # 前向传播
            with torch.set_grad_enabled(training_mode):
                if training_mode:
                    output = model(batch)
                    loss = output.sum()
                    loss.backward()
                else:
                    with torch.no_grad():
                        output = model(batch)
            
            peak_memory = self.profiler.get_memory_info()
            
            # 清理
            del model, batch, output
            if training_mode and 'loss' in locals():
                del loss
            
            return {
                'actual_edges': actual_edges,
                'peak_allocated': peak_memory['max_allocated'],
                'peak_reserved': peak_memory['max_reserved'],
                'memory_increase': peak_memory['max_allocated'] - initial_memory['allocated']
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                return {
                    'actual_edges': num_edges,
                    'peak_allocated': float('inf'),
                    'peak_reserved': float('inf'),
                    'memory_increase': float('inf'),
                    'oom_error': True
                }
            else:
                raise e


class MemoryAnalyzer:
    """显存分析主类"""
    
    def __init__(self, config: MemoryTestConfig):
        self.config = config
        self.tester = ComponentTester(config)
        self.results = {}
    
    def run_comprehensive_analysis(self) -> Dict[str, pd.DataFrame]:
        """运行全面的显存分析"""
        print("🚀 开始GAOT-3D显存分析...")
        print(f"📊 测试配置: {self.config.num_test_points}个edges规模点")
        print(f"🎯 Edges范围: {self.config.min_edges:,} - {self.config.max_edges:,}")
        print(f"🔧 设备: {self.config.device}")
        print(f"🧮 物理节点数: {self.config.num_physical_nodes:,}")
        print(f"🔮 潜在Tokens数: {self.config.num_latent_tokens:,}")
        print("="*80)
        
        # 生成测试的edges数量范围 (对数分布)
        edges_range = np.logspace(
            np.log10(self.config.min_edges),
            np.log10(self.config.max_edges),
            self.config.num_test_points
        ).astype(int)
        
        test_scenarios = []
        
        # 定义所有测试场景
        if self.config.test_encoder:
            if self.config.test_training_mode:
                test_scenarios.append(('encoder', 'training'))
            if self.config.test_inference_mode:
                test_scenarios.append(('encoder', 'inference'))
        
        if self.config.test_decoder:
            if self.config.test_training_mode:
                test_scenarios.append(('decoder', 'training'))
            if self.config.test_inference_mode:
                test_scenarios.append(('decoder', 'inference'))
        
        if self.config.test_full_model:
            if self.config.test_training_mode:
                test_scenarios.append(('full_model', 'training'))
            if self.config.test_inference_mode:
                test_scenarios.append(('full_model', 'inference'))
        
        # 执行所有测试
        for component, mode in test_scenarios:
            print(f"\n🧪 测试 {component.upper()} - {mode.upper()} 模式")
            
            results = []
            for i, num_edges in enumerate(edges_range):
                print(f"  📏 [{i+1:2d}/{len(edges_range)}] Edges: {num_edges:8,}", end=" -> ")
                
                # 根据组件类型选择测试函数
                if component == 'encoder':
                    result = self.tester.test_encoder_memory(num_edges, mode=='training')
                elif component == 'decoder':
                    result = self.tester.test_decoder_memory(num_edges, mode=='training')
                elif component == 'full_model':
                    result = self.tester.test_full_model_memory(num_edges, mode=='training')
                
                # 添加测试信息
                result.update({
                    'component': component,
                    'mode': mode,
                    'target_edges': num_edges
                })
                
                results.append(result)
                
                # 输出结果
                if 'oom_error' in result:
                    print("💥 OOM!")
                    break
                else:
                    print(f"💾 {result['peak_allocated']:.1f}MB")
                
                time.sleep(0.1)  # 短暂延迟让显存稳定
            
            # 保存结果
            key = f"{component}_{mode}"
            self.results[key] = pd.DataFrame(results)
        
        print("\n✅ 分析完成!")
        return self.results
    
    def generate_report(self) -> str:
        """生成分析报告"""
        report = []
        report.append("📊 GAOT-3D 显存分析报告")
        report.append("="*60)
        report.append("")
        
        for key, df in self.results.items():
            component, mode = key.split('_', 1)
            report.append(f"🔍 {component.upper()} - {mode.upper()}模式:")
            
            if df.empty:
                report.append("  ❌ 无数据")
                continue
            
            # 过滤掉OOM的结果 - 修复pandas DataFrame的使用
            if 'oom_error' in df.columns:
                valid_df = df[~df['oom_error'].fillna(False)]
            else:
                valid_df = df  # 如果没有oom_error列，说明没有OOM
            
            if valid_df.empty:
                report.append("  💥 所有测试都发生OOM")
                continue
            
            max_edges = valid_df['actual_edges'].max()
            max_memory = valid_df['peak_allocated'].max()
            min_memory = valid_df['peak_allocated'].min()
            
            report.append(f"  📏 最大成功edges数: {max_edges:,}")
            report.append(f"  💾 峰值显存: {max_memory:.1f}MB")
            report.append(f"  💚 最小显存: {min_memory:.1f}MB")
            report.append(f"  📈 显存增长: {max_memory/min_memory:.2f}x")
            report.append("")
        
        return "\n".join(report)
    
    def plot_results(self, save_path: str = "memory_analysis_results.png"):
        """绘制分析结果图表"""
        if not self.results:
            print("❌ 无结果数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for key, df in self.results.items():
            if df.empty or plot_idx >= 4:
                continue
            
            component, mode = key.split('_', 1)
            
            # 过滤有效数据 - 修复pandas DataFrame的使用
            if 'oom_error' in df.columns:
                valid_df = df[~df['oom_error'].fillna(False)]
            else:
                valid_df = df  # 如果没有oom_error列，说明没有OOM
            
            if valid_df.empty:
                continue
            
            ax = axes[plot_idx]
            
            # 绘制峰值显存 vs edges数量
            ax.loglog(valid_df['actual_edges'], valid_df['peak_allocated'], 
                     'o-', color=colors[plot_idx], linewidth=2, markersize=6,
                     label=f"{component.title()} ({mode})")
            
            ax.set_xlabel('边数量 (Edges)', fontsize=12, fontweight='bold')
            ax.set_ylabel('峰值显存 (MB)', fontsize=12, fontweight='bold')
            ax.set_title(f'{component.upper()} - {mode.upper()}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加拟合线
            if len(valid_df) > 3:
                try:
                    log_edges = np.log10(valid_df['actual_edges'])
                    log_memory = np.log10(valid_df['peak_allocated'])
                    
                    # 线性拟合 (在对数空间)
                    coeff = np.polyfit(log_edges, log_memory, 1)
                    
                    # 绘制拟合线
                    x_fit = np.logspace(np.log10(valid_df['actual_edges'].min()), 
                                      np.log10(valid_df['actual_edges'].max()), 100)
                    y_fit = 10**(coeff[0] * np.log10(x_fit) + coeff[1])
                    
                    ax.plot(x_fit, y_fit, '--', color=colors[plot_idx], alpha=0.7,
                           label=f'Slope: {coeff[0]:.2f}')
                    ax.legend()
                    
                except:
                    pass
            
            plot_idx += 1
        
        # 移除未使用的子图
        for i in range(plot_idx, 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 结果图表已保存至: {save_path}")
        plt.show()
    
    def save_raw_data(self, save_path: str = "memory_analysis_data.csv"):
        """保存原始分析数据"""
        if not self.results:
            print("❌ 无数据可保存")
            return
        
        # 合并所有结果
        all_data = []
        for key, df in self.results.items():
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(save_path, index=False)
            print(f"💾 原始数据已保存至: {save_path}")
        else:
            print("❌ 无有效数据可保存")


def main():
    """主函数"""
    print("🧠 GAOT-3D 显存分析工具")
    print("="*50)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用！请在GPU环境中运行此脚本")
        return
    
    # 显示GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 显存: {gpu_memory:.1f}GB")
    print()
    
    save_folder = "results/memory_analysis"
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "memory_analysis_results.png")

    # 创建测试配置
    config = MemoryTestConfig(
        device="cuda:0",
        batch_size=1,
        num_physical_nodes=500000,
        num_latent_tokens=4096,
        min_edges=500000,
        max_edges=5000000,  # 根据你的GPU调整
        num_test_points=12,
        test_encoder=True,
        test_decoder=True,
        test_full_model=True,  # 完整模型测试可能很慢
        test_training_mode=True,
        test_inference_mode=True
    )
    
    # 创建分析器并运行
    analyzer = MemoryAnalyzer(config)
    
    try:
        results = analyzer.run_comprehensive_analysis()
        
        # 生成报告
        report = analyzer.generate_report()
        print("\n" + report)
        
        # 绘制结果
        analyzer.plot_results(save_path=save_path)
        
        # 保存数据
        analyzer.save_raw_data(save_path=os.path.join(save_folder, "memory_analysis_data.csv"))
        
        print("\n🎉 分析完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断分析")
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 