#!/usr/bin/env python3
"""
Simplified Edge Memory Profiler: Specifically tests memory usage of MLP edge processing in IntegralTransform

Key Analysis:
1. Memory peak of channel_mlp edge processing in IntegralTransform.forward()
2. Memory variations under different MLP configurations 
3. Training vs inference mode comparison
4. Provides concise analysis and optimization suggestions

Usage:
python edge_memory_profiler.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.model.layers.integral_transform import IntegralTransform
    from src.model.layers.mlp import LinearChannelMLP
except ImportError as e:
    print(f"❌ Failed to import project modules: {e}")
    print("Please run this script from GAOT-3D project root directory")
    sys.exit(1)


class EdgeMemoryTester:
    """Edge Memory Tester"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA environment required")
        
        self.device_id = int(device.split(':')[1]) if ':' in device else 0
    
    def get_memory_mb(self) -> float:
        """Get current memory usage (MB)"""
        torch.cuda.synchronize(self.device_id)
        return torch.cuda.memory_allocated(self.device_id) / 1024**2
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage (MB)"""
        torch.cuda.synchronize(self.device_id)
        return torch.cuda.max_memory_allocated(self.device_id) / 1024**2
    
    def reset_memory_stats(self):
        """Reset memory statistics"""
        torch.cuda.reset_peak_memory_stats(self.device_id)
        torch.cuda.empty_cache()
        gc.collect()
    
    def test_integral_transform_memory(self, 
                                     num_edges: int,
                                     coord_dim: int = 3,
                                     feature_dim: int = 16,
                                     mlp_hidden_layers: List[int] = [64, 64, 64],
                                     training_mode: bool = True) -> Dict[str, float]:
        """
        Test memory usage of IntegralTransform with specified number of edges
        
        Args:
            num_edges: Number of edges
            coord_dim: Coordinate dimension
            feature_dim: Feature dimension
            mlp_hidden_layers: MLP hidden layer configuration
            training_mode: Whether in training mode
        """
        self.reset_memory_stats()
        
        try:
            # Calculate MLP input dimension: concatenated coordinates (source + target) + features
            mlp_input_dim = coord_dim * 2   # Assuming linear mode
            mlp_output_dim = feature_dim
            
            # Build complete MLP layer configuration
            full_mlp_layers = [mlp_input_dim] + mlp_hidden_layers + [mlp_output_dim]
            
            # Create IntegralTransform
            integral_transform = IntegralTransform(
                channel_mlp_layers=full_mlp_layers,
                transform_type="linear",
                use_attn=True,
                coord_dim=coord_dim,
                attention_type="cosine",
                sampling_strategy=None,
                sample_ratio= None
            ).to(self.device)
            
            if training_mode:
                integral_transform.train()
            else:
                integral_transform.eval()
            
            # Prepare test data
            num_source_nodes = max(1000, num_edges // 100)  # Dynamically set source nodes
            num_query_nodes = max(1000, num_edges // 100)   # Dynamically set query nodes
            
            # Generate coordinates
            source_pos = torch.randn(num_source_nodes, coord_dim, device=self.device)
            query_pos = torch.randn(num_query_nodes, coord_dim, device=self.device)
            
            # Generate features
            source_features = torch.randn(num_source_nodes, feature_dim, device=self.device)
            
            # Generate edges (random connections)
            actual_edges = min(num_edges, num_source_nodes * num_query_nodes)
            edge_index = torch.stack([
                torch.randint(0, num_query_nodes, (actual_edges,), device=self.device),
                torch.randint(0, num_source_nodes, (actual_edges,), device=self.device)
            ])
            
            initial_memory = self.get_memory_mb()
            
            # Test forward pass
            with torch.set_grad_enabled(training_mode):
                if training_mode:
                    # Training mode: need gradients
                    output = integral_transform(
                        y_pos=source_pos,
                        x_pos=query_pos,
                        edge_index=edge_index,
                        f_y=source_features
                    )
                    loss = output.sum()
                    loss.backward()
                else:
                    # Inference mode
                    with torch.no_grad():
                        output = integral_transform(
                            y_pos=source_pos,
                            x_pos=query_pos,
                            edge_index=edge_index,
                            f_y=source_features
                        )
            
            peak_memory = self.get_peak_memory_mb()
            memory_used = peak_memory - initial_memory
            
            # Calculate theoretical memory cost
            # MLP layers
            layers = [mlp_input_dim] + mlp_hidden_layers + [mlp_output_dim]
            
            # Calculate MLP parameter count (fixed overhead)
            total_mlp_params = 0
            for i in range(len(layers) - 1):
                total_mlp_params += layers[i] * layers[i+1] + layers[i+1]
            
            # Calculate total activation size (main cost that scales with edges)
            total_activation_size = sum(layers)  # Sum of activation sizes for all layers
            
            # Memory cost per edge is mainly activation values
            if training_mode:
                # Training: activation + gradient + optimizer states
                multiplier = 2.5  # Activation + gradient + optimizer overhead
            else:
                # Inference: mainly activation values
                multiplier = 1.0
            
            edge_related_memory = actual_edges * total_activation_size * 4 * multiplier / 1024**2
            
            # Cleanup
            del integral_transform, output
            if training_mode and 'loss' in locals():
                del loss
            
            return {
                'num_edges': actual_edges,
                'peak_memory_mb': peak_memory,
                'memory_used_mb': memory_used,
                'theoretical_edge_mb': edge_related_memory,
                'total_mlp_params': total_mlp_params,
                'total_activation_size': total_activation_size,
                'mlp_layers': layers,
                'training_multiplier': multiplier,
                'success': True
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Rough estimate for OOM cases
                mlp_input_dim = coord_dim * 2 + feature_dim
                layers = [mlp_input_dim] + mlp_hidden_layers + [feature_dim]
                total_activation_size = sum(layers)
                
                return {
                    'num_edges': num_edges,
                    'peak_memory_mb': float('inf'),
                    'memory_used_mb': float('inf'),
                    'theoretical_edge_mb': float('inf'),
                    'total_mlp_params': sum(mlp_hidden_layers),  # Rough estimate
                    'total_activation_size': total_activation_size,
                    'mlp_layers': layers,
                    'training_multiplier': 2.5 if training_mode else 1.0,
                    'success': False,
                    'error': 'OOM'
                }
            else:
                raise e
    
    def run_edge_scaling_test(self, 
                            edge_counts: List[int],
                            mlp_configs: List[List[int]] = None,
                            modes: List[str] = ['inference', 'training']) -> pd.DataFrame:
        """
        Run edge count scaling test
        
        Args:
            edge_counts: List of edge counts to test
            mlp_configs: List of MLP configurations
            modes: List of test modes
        """
        if mlp_configs is None:
            mlp_configs = [
                [64, 64, 64],           # Small MLP
                [128, 128, 128],    # Medium MLP
                [256, 256, 256],    # Large MLP
            ]
        
        results = []

        print("🔍 Starting Edge Scaling Test...")
        print(f"📏 Testing {len(edge_counts)} edge scales")
        print(f"🧮 Testing {len(mlp_configs)} MLP configs")
        print(f"🎭 Test modes: {modes}")
        print("="*60)
        
        for mlp_idx, mlp_config in enumerate(mlp_configs):
            print(f"\n🧠 MLP Config {mlp_idx+1}/{len(mlp_configs)}: {mlp_config}")
            
            for mode in modes:
                print(f"  🎭 {mode.upper()} Mode:")
                
                for i, num_edges in enumerate(edge_counts):
                    print(f"    📏 [{i+1:2d}/{len(edge_counts)}] {num_edges:8,} edges", end=" -> ")
                    
                    result = self.test_integral_transform_memory(
                        num_edges=num_edges,
                        mlp_hidden_layers=mlp_config,
                        training_mode=(mode == 'training')
                    )
                    
                    # Add test info
                    result.update({
                        'mlp_config': str(mlp_config),
                        'mode': mode,
                        'target_edges': num_edges
                    })
                    
                    results.append(result)
                    
                    # Output result
                    if result['success']:
                        print(f"💾 {result['memory_used_mb']:.1f}MB")
                    else:
                        print(f"💥 {result['error']}")
                        # Skip larger edge tests after OOM
                        break
                    
                    time.sleep(0.1)  # Let memory stabilize
        
        return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> str:
    """Analyze test results"""
    report = []
    report.append("📊 Edge Memory Analysis Report")
    report.append("="*50)
    report.append("")
    
    # Group analysis by MLP config and mode
    for (mlp_config, mode), group in df.groupby(['mlp_config', 'mode']):
        successful = group[group['success'] == True]
        
        if successful.empty:
            report.append(f"❌ {mlp_config} - {mode}: All OOM")
            continue
        
        max_edges = successful['num_edges'].max()
        min_memory = successful['memory_used_mb'].min()
        max_memory = successful['memory_used_mb'].max()
        
        # Calculate memory growth rate
        if len(successful) > 1:
            edges_ratio = successful['num_edges'].max() / successful['num_edges'].min()
            memory_ratio = max_memory / min_memory
            scaling_factor = np.log(memory_ratio) / np.log(edges_ratio)
        else:
            scaling_factor = np.nan
        
        # Calculate activation and parameter information
        mlp_params = successful.iloc[0]['total_mlp_params'] if 'total_mlp_params' in successful.columns else 0
        activation_size = successful.iloc[0]['total_activation_size'] if 'total_activation_size' in successful.columns else 0
        theoretical_memory = successful.iloc[-1]['theoretical_edge_mb'] if 'theoretical_edge_mb' in successful.columns else 0
        actual_memory = successful.iloc[-1]['memory_used_mb']
        efficiency = theoretical_memory / actual_memory * 100 if actual_memory > 0 else 0
        
        report.append(f"✅ {mlp_config} - {mode}:")
        report.append(f"   📏 max edges: {max_edges:,}")
        report.append(f"   🧠 MLP params: {mlp_params:,} (fixed)")
        report.append(f"   🔥 activation size: {activation_size} (per edge)")
        report.append(f"   💾 memory range: {min_memory:.1f} - {max_memory:.1f} MB")
        report.append(f"   📈 scaling factor: {scaling_factor:.2f}")
        report.append(f"   🎯 theoretical accuracy: {efficiency:.1f}%")
        report.append("")
    
    # Training vs inference comparison
    report.append("🎭 training vs inference mode comparison:")
    for mlp_config in df['mlp_config'].unique():
        mlp_data = df[df['mlp_config'] == mlp_config]
        train_data = mlp_data[mlp_data['mode'] == 'training']
        infer_data = mlp_data[mlp_data['mode'] == 'inference']
        
        if not train_data.empty and not infer_data.empty:
            # Find common edge counts
            common_edges = set(train_data['num_edges']) & set(infer_data['num_edges'])
            if common_edges:
                edge_count = max(common_edges)
                train_mem = train_data[train_data['num_edges'] == edge_count]['memory_used_mb'].iloc[0]
                infer_mem = infer_data[infer_data['num_edges'] == edge_count]['memory_used_mb'].iloc[0]
                ratio = train_mem / infer_mem if infer_mem > 0 else float('inf')
                
                report.append(f"   {mlp_config}: training/inference = {ratio:.2f}x")
    
    # GPU capacity estimation
    report.append("")
    report.append("🎮 GPU capacity estimation (max edges for different MLP configs):")
    report.append("")
    
    # GPU specification definition
    gpu_specs = {
        'RTX 4090': 24 * 1024 * 0.90,   # Available memory is about 90% of total capacity
        'A100 40GB': 40 * 1024 * 0.90,
        'A100 80GB': 80 * 1024 * 0.90
    }
    
    # Estimate GPU capacity for each MLP config
    successful_df = df[df['success'] == True]
    if not successful_df.empty:
        for mlp_config in successful_df['mlp_config'].unique():
            mlp_data = successful_df[successful_df['mlp_config'] == mlp_config]
            
            report.append(f"📊 {mlp_config}:")
            
            for mode in ['inference', 'training']:
                mode_data = mlp_data[mlp_data['mode'] == mode]
                if len(mode_data) < 2:
                    continue
                    
                # Use linear fitting to estimate memory vs edges relationship
                edges = mode_data['num_edges'].values
                memory = mode_data['memory_used_mb'].values
                
                # Simple linear fitting (log space)
                if len(edges) >= 2:
                    try:
                        log_edges = np.log(edges)
                        log_memory = np.log(memory)
                        slope, intercept = np.polyfit(log_edges, log_memory, 1)
                        
                        report.append(f"   {mode} mode:")
                        for gpu_name, max_memory_mb in gpu_specs.items():
                            # Estimate max edges: max_edges = exp((log(max_memory) - intercept) / slope)
                            max_log_memory = np.log(max_memory_mb)
                            estimated_log_edges = (max_log_memory - intercept) / slope
                            estimated_max_edges = int(np.exp(estimated_log_edges))
                            
                            # Format edges number
                            if estimated_max_edges >= 1000000:
                                edges_str = f"{estimated_max_edges/1000000:.1f}M"
                            elif estimated_max_edges >= 1000:
                                edges_str = f"{estimated_max_edges/1000:.0f}K"
                            else:
                                edges_str = f"{estimated_max_edges:,}"
                            
                            report.append(f"     {gpu_name}: ~{edges_str} edges")
                    except:
                        report.append(f"   {mode} mode: not enough data to estimate")
            
            report.append("")
    
    return "\n".join(report)


def plot_results(df: pd.DataFrame, save_path: str = "edge_memory_analysis.png"):
    """Plot analysis results"""
    successful_df = df[df['success'] == True]
    
    if successful_df.empty:
        print("❌ No successful data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # GPU memory capacity definition (considering actual available memory is about 90% of total capacity)
    gpu_specs = {
        'RTX 4090': {'total_gb': 24, 'usable_mb': 24 * 1024 * 0.90, 'color': '#FF6B6B'},
        'A100 40GB': {'total_gb': 40, 'usable_mb': 40 * 1024 * 0.90, 'color': '#FFA500'}, 
        'A100 80GB': {'total_gb': 80, 'usable_mb': 80 * 1024 * 0.90, 'color': '#32CD32'}
    }
    
    # Subplot 1: Memory usage for different MLP configs
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, (mlp_config, group) in enumerate(successful_df.groupby('mlp_config')):
        for mode in ['inference', 'training']:
            mode_data = group[group['mode'] == mode]
            if not mode_data.empty:
                linestyle = '-' if mode == 'inference' else '--'
                ax1.loglog(mode_data['num_edges'], mode_data['memory_used_mb'], 
                          linestyle, color=colors[i % len(colors)], linewidth=2,
                          label=f"{mlp_config} ({mode})")
    
    # Add GPU memory capacity horizontal line
    for gpu_name, spec in gpu_specs.items():
        ax1.axhline(y=spec['usable_mb'], color=spec['color'], linestyle=':', 
                   linewidth=2, alpha=0.7, label=f"{gpu_name} ({spec['total_gb']}GB)")
    
    ax1.set_xlabel('Edges', fontweight='bold')
    ax1.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax1.set_title('Memory Usage vs Edges (with GPU capacity)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Subplot 2: Theoretical vs actual memory comparison
    for mode in ['inference', 'training']:
        mode_data = successful_df[successful_df['mode'] == mode]
        if not mode_data.empty:
            marker = 'o' if mode == 'inference' else 's'
            ax2.loglog(mode_data['theoretical_edge_mb'], mode_data['memory_used_mb'],
                      marker, alpha=0.7, label=f'{mode} actual')
    
    # Add theoretical line
    max_theoretical = successful_df['theoretical_edge_mb'].max()
    theoretical_line = np.logspace(0, np.log10(max_theoretical), 100)
    ax2.loglog(theoretical_line, theoretical_line, 'k--', alpha=0.5, label='theoretical')
    
    ax2.set_xlabel('Theoretical Activation Memory (MB)', fontweight='bold')
    ax2.set_ylabel('Actual Peak Memory (MB)', fontweight='bold')
    ax2.set_title('Theoretical vs Actual Memory (Activation-based)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Results plot saved: {save_path}")
    plt.show()


def main():
    """Main function"""
    print("🧠 Edge Memory Profiler")
    print("="*40)
    
    # Check environment
    if not torch.cuda.is_available():
        print("❌ CUDA environment required")
        return
    
    device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_memory:.1f}GB")
    print()
    
    # Create tester
    tester = EdgeMemoryTester(device)
    
    # Define test parameters
    edge_counts = [
        10000, 50000, 
        100000, 200000, 500000, 1000000, 5000000, 10000000, 20000000
    ]
    
    mlp_configs = [
        [64, 64, 64],       # Light
        [128, 128, 128],    # Standard
        [256, 256, 256],    # Heavy
    ]
    
    modes = ['inference', 'training']
    
    save_folder = "results/edge_memory_analysis"
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "edge_memory_analysis.png")


    try:
        # Run tests
        results_df = tester.run_edge_scaling_test(
            edge_counts=edge_counts,
            mlp_configs=mlp_configs,
            modes=modes
        )
        
        # Analyze results
        report = analyze_results(results_df)
        print("\n" + report)
        
        # Plot results
        plot_results(results_df, save_path=save_path)
        
        # Save data
        results_df.to_csv(os.path.join(save_folder, "edge_memory_results.csv"), index=False)
        print("💾 Detailed data saved: edge_memory_results.csv")
        
        print("\n🎉 Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 