import torch 
from dataclasses import dataclass
from typing import Union, Tuple
def rescale(x:torch.Tensor, lims=(-1,1))->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        ND tensor
    
    Returns
    -------
    x_normalized: torch.Tensor
        ND tensor
    """
    return (x-x.min()) / (x.max()-x.min()) * (lims[1] - lims[0]) + lims[0]


@dataclass
class MeanStd:
    mean:torch.Tensor
    std:torch.Tensor

def normalize(x:torch.Tensor, mean=None, std=None, return_mean_std:bool=False
              )->Union[torch.Tensor, Tuple[torch.Tensor, MeanStd]]:
    """
    Parameters
    ----------
    x: torch.Tensor
        ND tensor
    mean: Optional[float]
        mean of the data
    std: Optional[float]
        standard deviation of the data
    
    Returns
    -------
    x_normalized: torch.Tensor
        1D tensor
    """
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    if return_mean_std:
        return (x - mean) / std, MeanStd(mean, std)
    else:
        return (x - mean) / std
