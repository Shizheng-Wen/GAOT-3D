import os
import pyvista as pv
from tqdm import tqdm

from torch_geometric.data import Data
import torch

def stack_pressure_data(base_path, order_file, max_num = None, pressure_key='p', output_dir='output'):
    """
    Process VTK files in the order specified by order_file, stack coordinates and pressure,
    and save to a NetCDF file.

    Parameters:
    -----------
    base_path : str
        Root directory containing VTK folders (e.g., '/cluster/work/math/camlab-data/drivaernet/PressureVTK').
    order_file : str
        Path to order.txt specifying the processing sequence.
    max_num : int, optional
        only processing top max_num lines in the order_file (default: "200") 
    pressure_key : str, optional
        Key for pressure in VTK point_data (default: pressure 'p').
    output_dir : str, optional
        Directory for the output file (default: 'output').

    Returns:
    --------
    None
    """

    folder_mapping = {
            'E_S_WWC_WM': ['E_S_WWC_WM'],
            'E_S_WW_WM': ['E_S_WW_WM'],
            'F_D_WM_WW': ['F_D_WM_WW_1', 'F_D_WM_WW_2', 'F_D_WM_WW_3', 'F_D_WM_WW_4',
                        'F_D_WM_WW_5', 'F_D_WM_WW_6', 'F_D_WM_WW_7', 'F_D_WM_WW_8'],
            'F_S_WWC_WM': ['F_S_WWC_WM'],
            'F_S_WWS_WM': ['F_S_WWS_WM'],
            'N_S_WWC_WM': ['N_S_WWC_WM'],
            'N_S_WWS_WM': ['N_S_WWS_WM'],
            'N_S_WW_WM': ['N_S_WW_WM'],
        }

    with open(order_file, 'r') as f:
        order_list = [line.strip() for line in f if line.strip()]
    if max_num is not None:
        order_list = order_list[:max_num]

    for line in tqdm(order_list, desc="Processing VTK files"):
        # Parse folder name and number (e.g., 'N_S_WW_WM_633' -> 'N_S_WW_WM', '633')
        parts = line.rsplit('_', 1)
        if len(parts) != 2:
            print(f"Invalid format in order.txt: {line}")
            continue
        folder_name, number = parts

        if folder_name not in folder_mapping:
            print(f"unknown model: {folder_name} in order.txt")
            continue
        possible_folders = folder_mapping[folder_name]

        file_path = None
        for folder in possible_folders:
            vtk_filename = f"{folder_name}_{number}.vtk"
            vtk_filepath = os.path.join(base_path, folder, vtk_filename)
            
            if os.path.exists(vtk_filepath):
                file_path = vtk_filepath
                break
        if file_path is None:
            print(f"File not found {folder_name}_{number}: {possible_folders}")
            continue
        
        mesh = pv.read(file_path)
        coords = mesh.points # (sample_size, 3)
        x = mesh.point_data[pressure_key] # (sample_size,)
        data = Data(
            pos = torch.tensor(coords, dtype=torch.float32), # (sample_size, 3)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1), # (sample_size, 3)
            )
        data.filename = f"{folder_name}_{number}"
        processed_dir = output_dir
        os.makedirs(processed_dir, exist_ok=True)
        save_path = os.path.join(processed_dir, f"{folder_name}_{number}.pt")
        torch.save(data, save_path)

if __name__ == '__main__':
    base_path = "/cluster/work/math/camlab-data/drivaernet/WallShearStressVTK_Updated"
    order_file = "/cluster/work/math/camlab-data/drivaernet/order.txt"
    pressure_key = "wallShearStress"  
    max_num = None
    output_dir = "/cluster/work/math/camlab-data/graphnpde/drivaernet/processed_pyg_stress"

    stack_pressure_data(
        base_path=base_path,
        order_file=order_file,
        max_num=max_num,
        pressure_key=pressure_key,
        output_dir=output_dir
    )