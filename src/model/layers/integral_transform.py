# src/model/layers/magno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
import importlib

from .mlp import LinearChannelMLP

from torch_geometric.utils import dropout_edge

# Use torch_scatter directly if installed and needed
try:
    import torch_scatter
    if hasattr(torch_scatter, 'segment_csr'):
         from torch_scatter import segment_csr as scatter_segment_csr
         HAS_SCATTER = True
    # Check for scatter_reduce or equivalent needed for softmax/aggregation
    elif hasattr(torch_scatter, 'scatter'):
         from torch_scatter import scatter
         HAS_SCATTER = True # Use scatter for sum, mean, max
    else:
        HAS_SCATTER = False
except ImportError:
    HAS_SCATTER = False

# Keep LinearChannelMLP (or import if moved)

# --- Native PyTorch segment_csr (Fallback if torch_scatter unavailable) ---
# This implementation needs careful adjustment for edge_index format
def segment_sum_native(src, index, dim_size):
     # Naive implementation, can be slow
     out = torch.zeros((dim_size,) + src.shape[1:], dtype=src.dtype, device=src.device)
     out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
     return out

def segment_mean_native(src, index, dim_size):
     # Naive implementation
     out_sum = segment_sum_native(src, index, dim_size)
     counts = torch.bincount(index, minlength=dim_size).unsqueeze(-1).clamp(min=1)
     return out_sum / counts

def segment_max_native(src, index, dim_size):
     # Naive implementation
     out = torch.full((dim_size,) + src.shape[1:], float('-inf'), dtype=src.dtype, device=src.device)
     # Use scatter with reduce='max' - requires PyTorch 1.12+ I believe
     # Fallback is harder, maybe loop? For now, assume scatter_reduce='max' works if scatter exists.
     if hasattr(torch, 'scatter_reduce_'):
         # Use the built-in scatter_reduce_ if available
         # Note the index needs expansion to match src dims for scatter_reduce_
         expanded_index = index.unsqueeze(-1).expand_as(src)
         out.scatter_reduce_(0, expanded_index, src, reduce="amax", include_self=False)
         out = torch.where(out == float('-inf'), 0.0, out) # Replace -inf with 0 if no neighbors
         return out
     else: # Very basic fallback if no scatter_reduce
          print("Warning: segment_max_native requires PyTorch 1.12+ or torch_scatter. Max values might be incorrect.")
          # Approximate with sum for now, or implement loop
          return segment_sum_native(src, index, dim_size)


# --- Choose scatter implementation ---
if HAS_SCATTER and hasattr(torch_scatter, 'scatter'):
     print("Using torch_scatter.scatter for aggregation.")
     scatter_sum = lambda src, index, dim_size: scatter(src, index, dim=0, dim_size=dim_size, reduce='sum')
     scatter_mean = lambda src, index, dim_size: scatter(src, index, dim=0, dim_size=dim_size, reduce='mean')
     scatter_max = lambda src, index, dim_size: scatter(src, index, dim=0, dim_size=dim_size, reduce='max')[0] # scatter_max returns values and argmax
elif HAS_SCATTER and hasattr(torch_scatter, 'segment_csr'):
     # segment_csr is less flexible (needs CSR format), prefer scatter if available
     print("Using torch_scatter.segment_csr for aggregation (less flexible).")
     # Need wrapper functions to adapt edge_index to CSR or use native below
     # For now, fall back to native if only segment_csr is found but scatter isn't
     print("Warning: torch_scatter.segment_csr found but not torch_scatter.scatter. Falling back to native PyTorch aggregation.")
     scatter_sum = segment_sum_native
     scatter_mean = segment_mean_native
     scatter_max = segment_max_native
else:
    print("Warning: torch_scatter not found or suitable functions missing. Using native PyTorch aggregation (potentially slow).")
    scatter_sum = segment_sum_native
    scatter_mean = segment_mean_native
    scatter_max = segment_max_native
# -----

class IntegralTransform(nn.Module):
    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_attn=None,
        coord_dim=None,
        attention_type='cosine',
        # Neighbor Sampling Params
        sampling_strategy: Optional[Literal['max_neighbors', 'ratio']] = None,
        max_neighbors: Optional[int] = None,
        sample_ratio: Optional[float] = None
    ):
        super().__init__()
        # parameters for attentional integral transform
        self.transform_type = transform_type
        self.use_attn = use_attn
        self.coord_dim = coord_dim  
        self.attention_type = attention_type
        # parameters for neighbor sampling
        self.sampling_strategy = sampling_strategy
        self.max_neighbors = max_neighbors
        self.sample_ratio = sample_ratio

        # Init MLP based on channel_mlp or channel_mlp_layers
        if channel_mlp is None:
             if channel_mlp_layers is None: raise ValueError("Need channel_mlp or layers")
             self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp
        
        # Initialize attention projections if needed
        if self.use_attn:
            if coord_dim is None:
                raise ValueError("coord_dim must be specified when use_attn is True")

            if self.attention_type == 'dot_product':
                attention_dim = 64 
                self.query_proj = nn.Linear(self.coord_dim, attention_dim)
                self.key_proj = nn.Linear(self.coord_dim, attention_dim)
                self.scaling_factor = 1.0 / (attention_dim ** 0.5)
            elif self.attention_type == 'cosine':
                pass
            else:
                raise ValueError(f"Invalid attention_type: {self.attention_type}. Must be 'cosine' or 'dot_product'.")

    def _apply_neighbor_sampling(
        self,
        edge_index: torch.Tensor,
        num_query_nodes: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Applies neighbor sampling based on the configured strategy."""
        num_total_original_edges = edge_index.shape[1]

        if num_query_nodes == 0 or num_total_original_edges == 0:
            return edge_index 

        # --- Strategy 1: Max Neighbors Per Node ---
        if self.sampling_strategy == 'max_neighbors':
            # This remains tricky to vectorize efficiently with edge_index.
            # Using a loop over nodes requiring sampling is often the clearest.
            # PyG's dropout_adj has ratio-based, not max-count based logic.
            print("Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'.")

            dest_nodes = edge_index[0] 
            counts = torch.bincount(dest_nodes, minlength=num_query_nodes)
            needs_sampling_mask = counts > self.max_neighbors

            if not torch.any(needs_sampling_mask):
                return edge_index 

            keep_mask = torch.ones(num_total_original_edges, dtype=torch.bool, device=device)
            queries_to_sample_idx = torch.where(needs_sampling_mask)[0]

            for i in queries_to_sample_idx:
                node_edge_mask = (dest_nodes == i)
                node_edge_indices = torch.where(node_edge_mask)[0]
                num_node_edges = len(node_edge_indices) 

                perm = torch.randperm(num_node_edges, device=device)[:self.max_neighbors]
                edges_to_keep_for_node = node_edge_indices[perm]
                # Update mask: keep only sampled edges for this node
                node_keep_mask = torch.zeros_like(node_edge_mask) 
                node_keep_mask[edges_to_keep_for_node] = True
                keep_mask[node_edge_mask] = node_keep_mask[node_edge_mask]

            sampled_edge_index = edge_index[:, keep_mask]
            return sampled_edge_index

        # --- Strategy 2: Global Ratio Sampling ---
        elif self.sampling_strategy == 'ratio':
            if self.sample_ratio >= 1.0:
                 return edge_index
            p_drop = 1.0 - self.sample_ratio
            sampled_edge_index, _ = dropout_edge(edge_index, p=p_drop, force_undirected=False, training=self.training)
            return sampled_edge_index

        else:
             raise ValueError(f"Invalid sampling strategy: {self.sampling_strategy}")

    def _segment_softmax_pyg(self, scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        """Applies softmax per segment based on index using torch_scatter."""
        scores_max = scatter_max(scores, index, dim=0, dim_size=dim_size)
        # Ensure scores_max is broadcastable back
        scores_max_expanded = scores_max[index]
        scores = scores - scores_max_expanded # Stable softmax
        exp_scores = torch.exp(scores)
        exp_sum = scatter_sum(exp_scores, index, dim=0, dim_size=dim_size)
        # Clamp sum to avoid division by zero
        exp_sum_clamped = torch.clamp(exp_sum, min=torch.finfo(exp_sum.dtype).tiny)
        exp_sum_expanded = exp_sum_clamped[index]
        attention_weights = exp_scores / exp_sum_expanded
        return attention_weights
    
    def forward(self, y_pos: torch.Tensor, 
                x_pos: torch.Tensor, 
                edge_index: torch.Tensor, 
                f_y: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None, 
                batch_y=None, 
                batch_x=None):
        """
        Compute kernel integral transform using PyG edge_index.

        Args:
            y_pos (Tensor): Source node coordinates [N_y, D] (or [TotalNodes_y, D] if batched).
            x_pos (Tensor): Query node coordinates [N_x, D] (or [TotalNodes_x, D] if batched).
            edge_index (Tensor): Edge index [2, NumEdges] where edge_index[0] indexes into x_pos (query)
                                 and edge_index[1] indexes into y_pos (source).
            f_y (Tensor, optional): Source node features [N_y, C_in] (or [TotalNodes_y, C_in] if batched).
            weights (Tensor, optional): Edge weights [NumEdges,]. Not typically volume weights here.
            batch_y (Tensor, optional): Batch assignment for y_pos nodes [TotalNodes_y].
            batch_x (Tensor, optional): Batch assignment for x_pos nodes [TotalNodes_x]. Required if using batching.
        """
        device = x_pos.device
        num_query_nodes = x_pos.shape[0]

        # --- Apply Neighbor Sampling ---
        if self.sampling_strategy is not None:
            sampled_edge_index = self._apply_neighbor_sampling(edge_index.to(device), num_query_nodes, device)
        else:
            sampled_edge_index = edge_index.to(device)

        num_sampled_edges = sampled_edge_index.shape[1]
        if num_sampled_edges == 0:
            # Handle no neighbors
            output_channels = self.channel_mlp.fcs[-1].out_channels
            output_shape = [num_query_nodes, output_channels] # Non-batched shape
            if batch_x is not None and f_y is not None and f_y.ndim == 3: # Check if input was batched
                 # This case is ambiguous - if f_y was [B, N, C], output should be?
                 # Let's assume f_y is [TotalNodes, C] for PyG batching
                 pass # Output shape remains [TotalNodes_x, C_out]
            return torch.zeros(output_shape, device=device, dtype=self.channel_mlp.fcs[-1].weight.dtype)
        # --- End Neighbor Sampling ---


        query_idx = sampled_edge_index[0]
        source_idx = sampled_edge_index[1]

        rep_features_pos = y_pos[source_idx]   # Source node coords [NumSampledEdges, D]
        self_features_pos = x_pos[query_idx]    # Query node coords [NumSampledEdges, D]

        in_features = None
        if f_y is not None:
             # Assume f_y is [TotalNodes_y, C_in]
             in_features = f_y[source_idx] # Source node features [NumSampledEdges, C_in]


        # --- Attention Logic ---
        attention_weights = None
        if self.use_attn:
            query_coords = self_features_pos[:, :self.coord_dim]
            key_coords = rep_features_pos[:, :self.coord_dim]
            if self.attention_type == 'dot_product':
                 query = self.query_proj(query_coords)
                 key = self.key_proj(key_coords)
                 attention_scores = torch.sum(query * key, dim=-1) * self.scaling_factor
            elif self.attention_type == 'cosine':
                 query_norm = F.normalize(query_coords, p=2, dim=-1)
                 key_norm = F.normalize(key_coords, p=2, dim=-1)
                 attention_scores = torch.sum(query_norm * key_norm, dim=-1)
            else:
                 raise ValueError(f"Invalid attention_type: {self.attention_type}")
            attention_weights = self._segment_softmax_pyg(attention_scores, query_idx, num_query_nodes)
        # --- End Attention Logic ---


        # Create aggregated features for MLP input
        agg_features = torch.cat([rep_features_pos, self_features_pos], dim=-1) # [NumSampledEdges, 2*D]

        if in_features is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        rep_features_transformed = self.channel_mlp(agg_features) # [NumSampledEdges, C_mlp_out]

        if in_features is not None and self.transform_type != "nonlinear_kernelonly":
             rep_features_transformed = rep_features_transformed * in_features

        if attention_weights is not None:
             rep_features_transformed = rep_features_transformed * attention_weights.unsqueeze(-1)

        # Apply edge weights if provided (e.g., distances, kernel values not from MLP)
        reduction = "sum" if (self.use_attn and attention_weights is not None) else "mean"
        if weights is not None:
             # Assume weights correspond to the *original* edges, need to index them
             # This requires weights to be passed if sampling is ratio-based
             # If sampling is max_neighbor, how to get weights?
             # Let's assume weights are edge features computed *after* sampling if needed,
             # or simply not used with this sampling structure for now.
             # If weights are per-edge attributes in PyG Data, index them: weights[sampled_edge_indices?]
             # For simplicity, ignoring external weights when sampling for now.
             # If volume weights per *source node* 'y' were intended, index them: vol_weights[source_idx]
             # vol_weights_y = weights[source_idx]
             # rep_features_transformed = vol_weights_y.unsqueeze(-1) * rep_features_transformed
             # reduction = "sum" # Usually sum if volume weights are used
             pass # Ignoring external weights for now when sampling


        # === Final Aggregation using torch_scatter ===
        # Aggregate features based on the query node index
        if reduction == 'mean':
             out_features = scatter_mean(rep_features_transformed, query_idx, dim=0, dim_size=num_query_nodes)
        elif reduction == 'sum':
             out_features = scatter_sum(rep_features_transformed, query_idx, dim=0, dim_size=num_query_nodes)
        else: # E.g., max
             out_features = scatter_max(rep_features_transformed, query_idx, dim=0, dim_size=num_query_nodes)

        return out_features



