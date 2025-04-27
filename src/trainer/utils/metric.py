import torch
import numpy as np
from ...data.dataset import Metadata
from typing import List, Dict, Union 

EPSILON = 1e-10

# --- Poseidon Metrics ---

def compute_batch_errors(gtr: torch.Tensor, prd: torch.Tensor, metadata: Metadata) -> torch.Tensor:
    """
    Compute the per-sample relative L1 errors per variable chunk for a batch.
    
    Args:
        gtr (torch.Tensor): Ground truth tensor with shape [batch_size, time, space, var]
        prd (torch.Tensor): Predicted tensor with shape [batch_size, time, space, var]
        metadata (Metadata): Dataset metadata including global_mean, global_std, and variable chunks
    
    Returns:
        torch.Tensor: Relative errors per sample per variable chunk, shape [batch_size, num_chunks]
    """
    # normalize the data
    active_vars = metadata.active_variables

    mean = torch.tensor(metadata.global_mean, device=gtr.device, dtype=gtr.dtype)[active_vars].reshape(1, 1, 1, -1)
    std = torch.tensor(metadata.global_std, device=gtr.device, dtype=gtr.dtype)[active_vars].reshape(1, 1, 1, -1)
    
    original_chunks = metadata.chunked_variables
    chunked_vars = [original_chunks[i] for i in active_vars]
    unique_chunks = sorted(set(chunked_vars))
    chunk_map = {old_chunk: new_chunk for new_chunk, old_chunk in enumerate(unique_chunks)}
    adjusted_chunks = [chunk_map[chunk] for chunk in chunked_vars]
    num_chunks = len(unique_chunks)

    chunks = torch.tensor(adjusted_chunks, device=gtr.device, dtype=torch.long)  # Shape: [var]

    gtr_norm = (gtr - mean) / std
    prd_norm = (prd - mean) / std

    # compute absolute errors and sum over the time and space dimensions
    abs_error = torch.abs(gtr_norm - prd_norm)  # Shape: [batch_size, time, space, var]
    error_sum = torch.sum(abs_error, dim=(1, 2))  # Shape: [batch_size, var]

    # sum errors per variable chunk
    chunks_expanded = chunks.unsqueeze(0).expand(error_sum.size(0), -1)  # Shape: [batch_size, var]
    error_per_chunk = torch.zeros(error_sum.size(0), num_chunks, device=gtr.device, dtype=error_sum.dtype)
    error_per_chunk.scatter_add_(1, chunks_expanded, error_sum)

    # compute sum of absolute values of the ground truth per chunk
    gtr_abs_sum = torch.sum(torch.abs(gtr_norm), dim=(1, 2))  # Shape: [batch_size, var]
    gtr_sum_per_chunk = torch.zeros(gtr_abs_sum.size(0), num_chunks, device=gtr.device, dtype=gtr_abs_sum.dtype)
    gtr_sum_per_chunk.scatter_add_(1, chunks_expanded, gtr_abs_sum)

    # compute relative errors per chunk
    relative_error_per_chunk = error_per_chunk / (gtr_sum_per_chunk + EPSILON) # Shape: [batch_size, num_chunks]

    return relative_error_per_chunk # Shape: [batch_size, num_chunks]
    
def compute_final_metric(all_relative_errors: torch.Tensor) -> float:
    """
    Compute the final metric from the accumulated relative errors.
    
    Args:
        all_relative_errors (torch.Tensor): Tensor of shape [num_samples, num_chunks]
        
    Returns:
        Metrics: An object containing the final relative L1 median error
    """
    # Compute the median over the sample axis for each chunk
    median_error_per_chunk = torch.median(all_relative_errors, dim=0)[0]  # Shape: [num_chunks]

    # Take the mean of the median errors across all chunks
    final_metric = torch.mean(median_error_per_chunk)
    
    return final_metric.item()


# --- General Metrics ---

def compute_general_metrics_batch(gtr: torch.Tensor, prd: torch.Tensor) -> Dict[str, float]:
    """
    Computes general regression metrics for a batch of predictions.

    Args:
        gtr: Ground truth tensor [batch_size, ...]. Assumes de-normalized.
        prd: Prediction tensor [batch_size, ...]. Assumes de-normalized.

    Returns:
        Dictionary containing batch-averaged metrics:
        'mse', 'mae', 'max_ae', 'rel_l1', 'rel_l2'.
    """
    assert gtr.shape == prd.shape, "Ground truth and prediction tensors must have the same shape."
    batch_size = gtr.shape[0]
    gtr_flat = gtr.view(batch_size, -1)
    prd_flat = prd.view(batch_size, -1)
    diff_flat = prd_flat - gtr_flat

    # MSE, MAE, Max_AE
    mse_batch = torch.mean(diff_flat ** 2).item()
    mae_batch = torch.mean(torch.abs(diff_flat)).item()
    max_ae_batch = torch.max(torch.abs(diff_flat)).item()

    # Relative L2 Error per sample
    norm_diff_l2 = torch.linalg.norm(diff_flat, ord=2, dim=1)
    norm_gtr_l2 = torch.linalg.norm(gtr_flat, ord=2, dim=1)
    rel_l2_batch = torch.mean(norm_diff_l2 / (norm_gtr_l2 + EPSILON)).item() * 100.0

    # Relative L1 Error per sample
    norm_diff_l1 = torch.linalg.norm(diff_flat, ord=1, dim=1)
    norm_gtr_l1 = torch.linalg.norm(gtr_flat, ord=1, dim=1)
    rel_l1_batch = torch.mean(norm_diff_l1 / (norm_gtr_l1 + EPSILON)).item() * 100.0 # As percentage

    return {
        'mse': mse_batch,
        'mae': mae_batch,
        'max_ae': max_ae_batch,
        'rel_l2': rel_l2_batch,
        'rel_l1': rel_l1_batch,
    }

def aggregate_general_metrics(batch_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates metrics computed over multiple batches.

    Args:
        batch_metrics_list: A list where each element is a dictionary
                             returned by compute_general_metrics_batch.

    Returns:
        Dictionary containing aggregated metrics over the entire dataset:
        'MSE', 'MAE', 'Max AE', 'Rel L2 Error (%)', 'Rel L1 Error (%)'.
    """
    if not batch_metrics_list:
        return {
            'MSE': 0.0, 'MAE': 0.0, 'Max AE': 0.0,
            'Rel L2 Error (%)': 0.0, 'Rel L1 Error (%)': 0.0
        }


    num_batches = len(batch_metrics_list)
    agg_mse = sum(m['mse'] for m in batch_metrics_list) / num_batches
    agg_mae = sum(m['mae'] for m in batch_metrics_list) / num_batches
    agg_rel_l2 = sum(m['rel_l2'] for m in batch_metrics_list) / num_batches
    agg_rel_l1 = sum(m['rel_l1'] for m in batch_metrics_list) / num_batches
    agg_max_ae = max(m['max_ae'] for m in batch_metrics_list)

    # Return with keys suitable for reporting
    return {
        'MSE': agg_mse,
        'MAE': agg_mae,
        'Max AE': agg_max_ae,
        'Rel L2 Error (%)': agg_rel_l2,
        'Rel L1 Error (%)': agg_rel_l1,
    }


# --- Drivaernet Metrics ---
def compute_drivaernet_metric(gtr_ls: List[torch.Tensor], prd_ls: List[torch.Tensor], metadata:Metadata):
    num_batches = len(gtr_ls)
    MEAN = torch.tensor(metadata.global_mean)
    STD = torch.tensor(metadata.global_std)
    all_metrics = []
    for idx in range(num_batches):
        result = {
            'MSE': 0,
            'MAE': 0,
            'RMSE': 0,
            'Max_Error': 0,
            'Rel_L2': 0,
            'Rel_L1': 0
        }

        gtr = gtr_ls[idx]
        prd = prd_ls[idx]
        gtr_norm = (gtr - MEAN)/STD
        prd_norm = (prd - MEAN)/STD
        # Convert to numpy
        gtr_norm = gtr_norm.cpu().numpy()
        prd_norm = prd_norm.cpu().numpy()

        mse = np.mean((gtr_norm - prd_norm) ** 2)
        mae = np.mean(np.abs(gtr_norm - prd_norm))
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(gtr_norm - prd_norm))
        rel_l2 = np.mean(np.linalg.norm(gtr_norm - prd_norm, axis=0) / 
                        np.linalg.norm(gtr_norm, axis=0))
        rel_l1 = np.mean(np.sum(np.abs(gtr_norm - prd_norm), axis=0) / 
                        np.sum(np.abs(gtr_norm), axis=0))
        result['MSE'] = mse
        result['MAE'] = mae
        result['RMSE'] = rmse
        result['Max_Error'] = max_error
        result['Rel_L2'] = rel_l2
        result['Rel_L1'] = rel_l1
        all_metrics.append(result)
    agg_metrics = {}
    for metric_name in all_metrics[0].keys():
        agg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
        agg_metrics[f"{metric_name}_std"] = np.std([m[metric_name] for m in all_metrics])
    
    return agg_metrics


