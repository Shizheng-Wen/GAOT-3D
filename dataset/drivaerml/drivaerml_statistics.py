import os
import numpy as np
import pandas as pd
import torch
import pyvista as pv
from tqdm import tqdm
import argparse

def analyze_processed_data(data_dir, output_csv=None):
    """
    分析已处理的.pt文件，统计坐标范围和点云数目
    
    Parameters:
    -----------
    data_dir : str
        包含.pt文件的目录路径
    output_csv : str, optional
        输出CSV文件路径
    
    Returns:
    --------
    pd.DataFrame : 统计结果数据框
    """
    
    # 获取所有.pt文件
    pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    pt_files.sort()
    
    if not pt_files:
        print(f"在目录 {data_dir} 中未找到.pt文件")
        return None
    
    print(f"找到 {len(pt_files)} 个.pt文件")
    
    results = []
    
    for pt_file in tqdm(pt_files, desc="分析.pt文件"):
        file_path = os.path.join(data_dir, pt_file)
        
        try:
            # 读取.pt文件
            data = torch.load(file_path, map_location='cpu')
            
            # 获取坐标
            pos = data.pos.numpy()  # (num_points, 3)
            
            # 统计信息
            num_points = pos.shape[0]
            x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
            y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
            z_min, z_max = pos[:, 2].min(), pos[:, 2].max()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            results.append({
                'filename': pt_file.replace('.pt', ''),
                'num_points': num_points,
                'x_min': x_min,
                'x_max': x_max,
                'x_range': x_range,
                'y_min': y_min,
                'y_max': y_max,
                'y_range': y_range,
                'z_min': z_min,
                'z_max': z_max,
                'z_range': z_range
            })
            
        except Exception as e:
            print(f"处理文件 {pt_file} 时出错: {str(e)}")
            continue
    
    # 创建数据框
    df = pd.DataFrame(results)
    
    if not df.empty:
        # 打印总体统计
        print("\n=== 数据集总体统计 ===")
        print(f"总文件数: {len(df)}")
        print(f"点云数目统计:")
        print(f"  平均: {df['num_points'].mean():.1f}")
        print(f"  中位数: {df['num_points'].median():.1f}")
        print(f"  最小: {df['num_points'].min()}")
        print(f"  最大: {df['num_points'].max()}")
        
        print(f"\nX轴坐标统计:")
        print(f"  全局范围: [{df['x_min'].min():.6f}, {df['x_max'].max():.6f}]")
        print(f"  平均范围: {df['x_range'].mean():.6f}")
        
        print(f"\nY轴坐标统计:")
        print(f"  全局范围: [{df['y_min'].min():.6f}, {df['y_max'].max():.6f}]")
        print(f"  平均范围: {df['y_range'].mean():.6f}")
        
        print(f"\nZ轴坐标统计:")
        print(f"  全局范围: [{df['z_min'].min():.6f}, {df['z_max'].max():.6f}]")
        print(f"  平均范围: {df['z_range'].mean():.6f}")
    
    # 保存到CSV文件
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n统计结果已保存到: {output_csv}")
    
    return df

def analyze_raw_data(data_dir, output_csv=None, max_num=None):
    """
    分析原始.vtp文件，统计坐标范围和点云数目
    
    Parameters:
    -----------
    data_dir : str
        包含.vtp文件的目录路径
    output_csv : str, optional
        输出CSV文件路径
    max_num : int, optional
        最大处理文件数
    
    Returns:
    --------
    pd.DataFrame : 统计结果数据框
    """
    
    # 获取所有.vtp文件
    vtp_files = [f for f in os.listdir(data_dir) if f.endswith('.vtp')]
    vtp_files.sort()
    
    if max_num is not None:
        vtp_files = vtp_files[:max_num]
    
    if not vtp_files:
        print(f"在目录 {data_dir} 中未找到.vtp文件")
        return None
    
    print(f"找到 {len(vtp_files)} 个.vtp文件")
    
    results = []
    
    for vtp_file in tqdm(vtp_files, desc="分析.vtp文件"):
        file_path = os.path.join(data_dir, vtp_file)
        
        try:
            # 读取VTP文件
            mesh = pv.read(file_path)
            
            # 转换cell data到point data
            mesh_with_point_data = mesh.cell_data_to_point_data()
            
            # 获取坐标
            coords = mesh_with_point_data.points  # (num_points, 3)
            
            # 统计信息
            num_points = coords.shape[0]
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            results.append({
                'filename': vtp_file.replace('.vtp', ''),
                'num_points': num_points,
                'x_min': x_min,
                'x_max': x_max,
                'x_range': x_range,
                'y_min': y_min,
                'y_max': y_max,
                'y_range': y_range,
                'z_min': z_min,
                'z_max': z_max,
                'z_range': z_range
            })
            
        except Exception as e:
            print(f"处理文件 {vtp_file} 时出错: {str(e)}")
            continue
    
    # 创建数据框
    df = pd.DataFrame(results)
    
    if not df.empty:
        # 打印总体统计
        print("\n=== 数据集总体统计 ===")
        print(f"总文件数: {len(df)}")
        print(f"点云数目统计:")
        print(f"  平均: {df['num_points'].mean():.1f}")
        print(f"  中位数: {df['num_points'].median():.1f}")
        print(f"  最小: {df['num_points'].min()}")
        print(f"  最大: {df['num_points'].max()}")
        
        print(f"\nX轴坐标统计:")
        print(f"  全局范围: [{df['x_min'].min():.6f}, {df['x_max'].max():.6f}]")
        print(f"  平均范围: {df['x_range'].mean():.6f}")
        
        print(f"\nY轴坐标统计:")
        print(f"  全局范围: [{df['y_min'].min():.6f}, {df['y_max'].max():.6f}]")
        print(f"  平均范围: {df['y_range'].mean():.6f}")
        
        print(f"\nZ轴坐标统计:")
        print(f"  全局范围: [{df['z_min'].min():.6f}, {df['z_max'].max():.6f}]")
        print(f"  平均范围: {df['z_range'].mean():.6f}")
    
    # 保存到CSV文件
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n统计结果已保存到: {output_csv}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="统计drivaerml数据的坐标范围和点云数目")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='数据目录路径 (包含.pt或.vtp文件)')
    parser.add_argument('--data_type', type=str, choices=['pt', 'vtp'], default='vtp',
                       help='数据类型: pt (处理后的PyTorch文件) 或 vtp (原始VTP文件)')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='输出CSV文件路径 (可选)')
    parser.add_argument('--max_num', type=int, default=None,
                       help='最大处理文件数 (仅对vtp有效)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"错误: 目录 {args.data_dir} 不存在")
        return
    
    print(f"分析目录: {args.data_dir}")
    print(f"数据类型: {args.data_type}")
    
    if args.data_type == 'pt':
        df = analyze_processed_data(args.data_dir, args.output_csv)
    else:
        df = analyze_raw_data(args.data_dir, args.output_csv, args.max_num)
    
    if df is not None and not df.empty:
        print(f"\n前5个文件的详细统计:")
        print(df.head().to_string(index=False))
    else:
        print("未能生成统计结果")

if __name__ == '__main__':
    # 如果直接运行脚本，使用默认配置
    if len(os.sys.argv) == 1:
        # 默认配置，分析处理后的数据
        processed_data_dir = "/cluster/work/math/camlab-data/drivaerml/surface"
        output_csv = "drivaerml_statistics.csv"
        
        print("使用默认配置分析处理后的数据...")
        df = analyze_processed_data(processed_data_dir, output_csv)
        
        if df is not None and not df.empty:
            print(f"\n前10个文件的详细统计:")
            print(df.head(10).to_string(index=False))
    else:
        main() 