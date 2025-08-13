import os
import xarray as xr
from tqdm import tqdm
import torch
from torch_geometric.data import Data

def process_nasa_crm_nc(nc_file, order_file, output_dir='processed_pyg'):
    """
    Process NASA CRM NetCDF files and convert each sample to PyTorch Geometric Data format
    
    Parameters:
    -----------
    nc_file : str
        Path to NetCDF file
    order_file : str  
        Path to text file containing sample name order
    output_dir : str, optional
        Output directory (default: 'processed_pyg')
        
    Returns:
    --------
    None
    """
    
    # Read NetCDF data
    print("Reading NetCDF file...")
    dataset = xr.open_dataset(nc_file)
    
    # Read order file
    with open(order_file, 'r') as f:
        order_list = [line.strip() for line in f if line.strip()]
    
    # Check if sample counts match
    n_samples = dataset.dims['samples']
    if len(order_list) != n_samples:
        print(f"Warning: Number of samples in order_file ({len(order_list)}) does not match number in NetCDF ({n_samples})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting to process {min(len(order_list), n_samples)} samples...")
    
    # Process each sample
    for i in tqdm(range(min(len(order_list), n_samples)), desc="Processing samples"):
        sample_name = order_list[i]
        
        # Extract data for current sample (remove time dimension, assume time=0)
        pos = dataset.x.isel(samples=i, time=0).values  # (points, x_channels)
        u = dataset.u.isel(samples=i, time=0).values    # (points, u_channels) 
        c = dataset.c.isel(samples=i, time=0).values    # (points, c_channels)
        
        # Create Data object
        data = Data(
            pos=torch.tensor(pos, dtype=torch.float32),  # (points, x_channels)
            x=torch.tensor(u, dtype=torch.float32),      # (points, u_channels)
            c=torch.tensor(c, dtype=torch.float32)       # (points, c_channels)
        )
        
        # Add filename information
        data.filename = sample_name
        
        # Save as .pt file
        save_path = os.path.join(output_dir, f"{sample_name}.pt")
        torch.save(data, save_path)
    
    # Close dataset
    dataset.close()
    print(f"Processing complete! All files saved to {output_dir}")

if __name__ == '__main__':
    # Configure paths
    nc_file = "/cluster/work/math/camlab-data/graphnpde/nasa_crm/nasa_crm.nc"
    order_file = "/cluster/work/math/camlab-data/graphnpde/nasa_crm/order_use.txt"
    output_dir = "/cluster/work/math/camlab-data/graphnpde/nasa_crm/processed_pyg"
    
    process_nasa_crm_nc(
        nc_file=nc_file,
        order_file=order_file, 
        output_dir=output_dir
    ) 