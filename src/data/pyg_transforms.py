# src/trainer/utils/pyg_transforms.py

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

try: 
    from src.utils.scale import rescale, normalize, rescale_new
    EPSILON = 1e-10
except ImportError:
    print("Error: Cannot import rescale/normalize from src.utils.scale")
    def rescale(x, lims=(-1,1)): return x
    def normalize(x, mean, std, return_mean_std=False): return x / (std + 1e-8)
    EPSILON = 1e-10

class RescalePosition(BaseTransform):
    """Rescales node positions 'pos' to a specified range (default: [-1, 1])."""
    def __init__(self, lims=(-1., 1.)):
        self.lims = lims
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = rescale(data.pos, lims=self.lims)
        else:
            print("Warning: RescalePosition transform called but data has no 'pos' attribute.")
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(lims={self.lims})'


class RescalePositionNew(BaseTransform):
    """Rescales node positions 'pos' to a specified range (default: [-1, 1])."""
    def __init__(self, lims=(-1., 1.), phy_domain = ([-1.16, -1.2, 0.0], [4.21, 1.19, 1.77])):
        self.lims = lims
        self.phy_domain = phy_domain
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = rescale_new(data.pos, lims=self.lims, phys_domain=self.phy_domain)
        else:
            print("Warning: RescalePosition transform called but data has no 'pos' attribute.")
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(lims={self.lims})'


class NormalizeFeatures(BaseTransform):
    """Normalizes node features 'x' and optionally 'c' using pre-computed mean and std."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, c_mean: torch.Tensor = None, c_std: torch.Tensor = None):
        self.mean = mean.detach()
        self.std = std.detach()
        
        self.c_mean = c_mean.detach() if c_mean is not None else None
        self.c_std = c_std.detach() if c_std is not None else None

    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'x') and data.x is not None:
            mean_dev = self.mean.to(data.x.device)
            std_dev = self.std.to(data.x.device)
            data.x = (data.x - mean_dev) / (std_dev + EPSILON) # Add epsilon for safety
        else:
            print("Warning: NormalizeFeatures transform called but data has no 'x' attribute.")
        
        if hasattr(data, 'c') and data.c is not None and self.c_mean is not None and self.c_std is not None:
            c_mean_dev = self.c_mean.to(data.c.device)
            c_std_dev = self.c_std.to(data.c.device)
            data.c = (data.c - c_mean_dev) / (c_std_dev + EPSILON) # Add epsilon for safety
            
        return data

    def __repr__(self) -> str:
        has_c = self.c_mean is not None and self.c_std is not None
        return f'{self.__class__.__name__}(mean=..., std=..., has_c_norm={has_c})'


