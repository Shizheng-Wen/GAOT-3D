import torch
from typing import Literal, Optional, Tuple

def scatter_native(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                    out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
                    reduce: str = "sum") -> torch.Tensor:
    if dim != 0:
        raise NotImplementedError("Native scatter fallback only supports dim=0")
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    shape = list(src.shape)
    shape[dim] = dim_size
    if out is None:
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    else:
        out = out.fill_(0.0) # Ensure out is zeroed

    index_expanded = index.view([-1] + [1] * (src.dim() - 1)).expand_as(src)

    if reduce == "sum" or reduce == "add":
        return out.scatter_add_(dim, index_expanded, src)
    elif reduce == "mean":
        out_sum = out.scatter_add_(dim, index_expanded, src)
        # Count occurrences, ensuring correct shape for broadcasting
        counts = torch.bincount(index, minlength=dim_size).to(src.dtype).to(src.device)
        # Add dimensions for broadcasting to match out_sum shape
        count_shape = [1] * src.dim()
        count_shape[dim] = dim_size
        counts = counts.view(count_shape)
        return out_sum / counts.clamp(min=1)
    elif reduce == "max" or reduce == "amax":
        # scatter_reduce_ supports 'amax'
            if hasattr(out, 'scatter_reduce_'):
                # Use 'include_self=False' as scatter_add_ already zeroed the output
                out.scatter_reduce_(dim, index_expanded, src, reduce="amax", include_self=False)
                # Replace remaining -inf (from initialization if used) or check zeros
                # If using scatter_add_ zero init, max will be >= 0
                return out
            else: # Very basic fallback if no scatter_reduce
                print("Warning: Native 'max' reduction is approximate without scatter_reduce_.")
                # Fallback: return sum as a placeholder, proper max needs more complex logic or looping
                return out.scatter_add_(dim, index_expanded, src)
    elif reduce == "min" or reduce == "amin":
        if hasattr(out, 'scatter_reduce_'):
                out.fill_(float('inf')) # Initialize for min
                out.scatter_reduce_(dim, index_expanded, src, reduce="amin", include_self=False)
                out = torch.where(out == float('inf'), 0.0, out) # Replace inf with 0 if no neighbors
                return out
        else:
                print("Warning: Native 'min' reduction not fully implemented without scatter_reduce_.")
                return out.scatter_add_(dim, index_expanded, src) # Placeholder
    else:
        raise ValueError(f"Unsupported reduce operation '{reduce}' in native scatter")
