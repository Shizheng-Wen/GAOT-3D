import os 
import pyvista as pv
from tqdm import tqdm

from torch_geometric.data import Data
import torch

def process_drivaerml_surface(base_path, output_dir, max_num=None, keys='pMeanTrim'):
    """
    Process VTP files from drivaerml surface data
    
    Parameters:
    -----------
    base_path : str
        Path to directory containing VTP files (e.g., '/cluster/work/math/camlab-data/drivaerml/surface')
    output_dir : str
        Directory for the output .pt files
    max_num : int, optional
        Maximum number of files to process (default: None, process all)
    keys : str or list, optional
        Key(s) for data in VTP cell_data to extract (default: 'pMeanTrim')
        
    Returns:
    --------
    None
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all VTP files in the directory
    vtp_files = [f for f in os.listdir(base_path) if f.endswith('.vtp')]
    vtp_files.sort()  # Sort for consistent processing order
    
    if max_num is not None:
        vtp_files = vtp_files[:max_num]
    
    print(f"Found {len(vtp_files)} VTP files to process")
    
    # Ensure keys is a list
    if isinstance(keys, str):
        keys = [keys]
    
    for vtp_file in tqdm(vtp_files, desc="Processing VTP files"):
        file_path = os.path.join(base_path, vtp_file)
        
        try:
            # Read VTP file
            mesh = pv.read(file_path)
            
            # Convert cell data to point data (interpolation)
            mesh_with_point_data = mesh.cell_data_to_point_data()
            
            # Get coordinates
            coords = mesh_with_point_data.points  # (num_points, 3)
            
            # Extract data for specified keys
            data_arrays = []
            for key in keys:
                if key in mesh_with_point_data.point_data:
                    data_array = mesh_with_point_data.point_data[key]
                    # Handle both scalar and vector data
                    if len(data_array.shape) == 1:
                        data_array = data_array.reshape(-1, 1)
                    data_arrays.append(data_array)
                else:
                    print(f"Warning: Key '{key}' not found in {vtp_file}")
                    continue
            
            if not data_arrays:
                print(f"No valid data found in {vtp_file}, skipping...")
                continue
            
            # Concatenate all data arrays
            if len(data_arrays) == 1:
                x = data_arrays[0]
            else:
                x = torch.cat([torch.tensor(arr, dtype=torch.float32) for arr in data_arrays], dim=1)
            
            # Create PyTorch Geometric Data object
            data = Data(
                pos=torch.tensor(coords, dtype=torch.float32),  # (num_points, 3)
                x=torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x  # (num_points, num_features)
            )
            
            # Set filename for reference
            data.filename = vtp_file.replace('.vtp', '')
            
            # Save as .pt file
            save_path = os.path.join(output_dir, vtp_file.replace('.vtp', '.pt'))
            torch.save(data, save_path)
            
        except Exception as e:
            print(f"Error processing {vtp_file}: {str(e)}")
            continue
    
    print(f"Processing complete. Files saved to {output_dir}")

if __name__ == '__main__':
    # Configuration
    base_path = "/cluster/work/math/camlab-data/drivaerml/surface"
    output_dir = "/cluster/work/math/camlab-data/graphnpde/drivaerml/processed_pyg_p"
    
    # You can specify multiple keys if needed
    # Available keys: 'CpMeanTrim', 'pMeanTrim', 'pPrime2MeanTrim', 'wallShearStressMeanTrim'
    keys = 'pMeanTrim'  # or ['pMeanTrim', 'CpMeanTrim'] for multiple keys
    
    max_num = None  # Process all files, or set to a number to limit
    
    process_drivaerml_surface(
        base_path=base_path,
        output_dir=output_dir,
        max_num=max_num,
        keys=keys
    )