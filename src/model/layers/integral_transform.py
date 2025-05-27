# src/model/layers/integral_transform.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
import importlib

from .mlp import LinearChannelMLP

from torch_geometric.utils import dropout_edge

# --- torch_scatter check ---
try:
    import torch_scatter
    # Check specifically for the 'scatter' function
    if hasattr(torch_scatter, 'scatter'):
        scatter = torch_scatter.scatter
        HAS_TORCH_SCATTER = True
        print("Using torch_scatter.scatter for aggregation.")
    else:
        HAS_TORCH_SCATTER = False
except ImportError:
    HAS_TORCH_SCATTER = False

if not HAS_TORCH_SCATTER:
    print("Warning: torch_scatter.scatter not found. Using native PyTorch fallbacks (potentially slower).")

    from .utils.scatter_native import scatter_native
    scatter = scatter_native 

"""
Reference: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/integral_transform.py
"""

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

        if sampling_strategy == 'max_neighbors':
            print("Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'.")

        # Init MLP based on channel_mlp or channel_mlp_layers
        if channel_mlp is None:
             if channel_mlp_layers is None: raise ValueError("Need channel_mlp or layers")
             self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp
        
        # InitiWarning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'.alize attention projections if needed
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
            # print("Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'.")

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
        scores_max = scatter(scores, index, dim=0, dim_size=dim_size, reduce="max")
        scores_max_expanded = scores_max[index]
        scores = scores - scores_max_expanded # Stable softmax
        exp_scores = torch.exp(scores)
        exp_sum = scatter(exp_scores, index, dim=0, dim_size=dim_size, reduce="sum")
        exp_sum_clamped = torch.clamp(exp_sum, min=torch.finfo(exp_sum.dtype).tiny) # Clamp sum to avoid division by zero
        exp_sum_expanded = exp_sum_clamped[index]
        attention_weights = exp_scores / exp_sum_expanded
        return attention_weights
    
    def forward(self, 
                y_pos: torch.Tensor, 
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
            output_channels = self.channel_mlp.fcs[-1].out_features
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
       
        out_features = scatter(
            src=rep_features_transformed,
            index=query_idx.long(),
            dim=0,                   # Aggregate along the edge dimension
            dim_size=num_query_nodes,# Output size is number of query nodes
            reduce=reduction
        )
        # Handle 'max'/'min' return tuple if using torch_scatter.scatter
        if HAS_TORCH_SCATTER and (reduction == "max" or reduction=="min"):
             out_features = out_features[0] # Keep only the values, discard argmax/argmin

        return out_features



