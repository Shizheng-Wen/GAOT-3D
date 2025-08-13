#src/model/layers/geoembed.py
import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

try:
    import torch_scatter
    if hasattr(torch_scatter, 'scatter'):
        scatter = torch_scatter.scatter
        HAS_TORCH_SCATTER = True
    else:
        HAS_TORCH_SCATTER = False
except ImportError:
    HAS_TORCH_SCATTER = False

if not HAS_TORCH_SCATTER:
    print("Warning: torch_scatter.scatter not found. Using native PyTorch fallbacks (potentially slower).")

    from .utils.scatter_native import scatter_native
    scatter = scatter_native


class GeometricEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, method='statistical', pooling='max', **kwargs):
        super(GeometricEmbedding, self).__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.method = method.lower()
        self.pooling = pooling.lower()
        self.kwargs = kwargs

        if self.pooling not in ['max', 'mean']:
            raise ValueError(f"Unsupported pooling method: {self.pooling}. Supported methods: 'max', 'mean'.")

        if self.method == 'statistical':
            # Feature dim: N_i, D_avg, D_var, Delta (D dims), PCA (D dims) = 3 + 2*D
            self.mlp = nn.Sequential(
                nn.Linear(self._get_stat_feature_dim(), 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )
        elif self.method == 'pointnet':
            # PointNet MLPs process individual centered neighbor coords
            self.pointnet_mlp = nn.Sequential(
                nn.Linear(input_dim, 32), 
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
            # FC layer after pooling
            self.fc = nn.Sequential(
                nn.Linear(32, output_dim), 
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forward(self,
                source_pos: torch.Tensor,
                query_pos: torch.Tensor,
                edge_index: torch.Tensor,
                batch_source: Optional[torch.Tensor] = None, 
                batch_query: Optional[torch.Tensor] = None,
                neighbors_counts: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Compute geometric embeddings using PyG batch format.

        Args:
            source_pos (Tensor): Coords of source nodes providing geometry [TotalSourceNodes, D].
            query_pos (Tensor): Coords of query nodes for which embeddings are computed [TotalQueryNodes, D].
            edge_index (Tensor): Bipartite edges [2, NumEdges], where edge_index[1] indexes query_pos,
                                 and edge_index[0] indexes source_pos.
            batch_source (Tensor, optional): Batch index for source nodes.
            batch_query (Tensor, optional): Batch index for query nodes.
            neighbors_counts (Tensor, optional): Number of neighbors for each query node.

        Returns:
            Tensor: Geometric embeddings for query nodes [TotalQueryNodes, output_dim].
        """
        if self.method == 'statistical':
            geo_features = self._compute_statistical_features_pyg(
                source_pos, query_pos, edge_index, neighbors_counts
            )
            return self.mlp(geo_features) 

        elif self.method == 'pointnet':
            geo_features = self._compute_pointnet_features_pyg(
                source_pos, query_pos, edge_index
            )
            return geo_features 

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _get_stat_feature_dim(self):
        # N_i, D_avg, D_var, Delta (D dims), PCA eigenvalues (D dims)
        return 3 + 2 * self.input_dim

    def _compute_statistical_features_pyg(self, source_pos, query_pos, edge_index, neighbors_counts = None):
        """
        Computes statistical geometric features using PyG edge_index.

        Parameters:
            source_pos (Tensor): Coords of source nodes providing geometry [TotalSourceNodes, D].
            query_pos (Tensor): Coords of query nodes for which embeddings are computed [TotalQueryNodes, D].
            edge_index (Tensor): Bipartite edges [2, NumEdges], where edge_index[1] indexes query_pos,
                                and edge_index[0] indexes source_pos.
            neighbors_counts (Tensor, optional): Number of neighbors for each query node.

        Returns:
            geo_features_normalized (torch.FloatTensor): The normalized geometric features, shape: [TotalQueryNodes, num_features]
        """
        num_queries = query_pos.shape[0]
        num_dims = query_pos.shape[1]
        device = query_pos.device
    
        neighbors_index = edge_index[0].long()             # Source node indices, shape: [NumEdges]
        query_indices_per_neighbor = edge_index[1].long()  # Query node indices for each neighbor, shape: [NumEdges]
        if neighbors_counts is None:
            num_neighbors_per_query = torch.bincount(query_indices_per_neighbor, minlength=num_queries).to(device)  # Number of neighbors per query node, shape: [num_queries]
        else:
            neighbors_counts = neighbors_counts.to(device)
            num_neighbors_per_query = neighbors_counts

        N_i = num_neighbors_per_query.float()  # Shape: [num_queries]
        has_neighbors = N_i > 0  # Shape: [num_queries]


        nbr_coords = source_pos[neighbors_index]  # Shape: [NumEdges, num_dims]
        query_coords_per_neighbor = query_pos[query_indices_per_neighbor]  # Shape: [NumEdges, num_dims]

        distances = torch.norm(nbr_coords - query_coords_per_neighbor, dim=1)  # Shape: [NumEdges]
        D_avg = scatter(distances, query_indices_per_neighbor, dim=0, dim_size=num_queries, reduce="mean")  # Average distance, shape: [num_queries]

        distances_squared = distances ** 2
        E_X2 = scatter(distances_squared, query_indices_per_neighbor, dim=0, dim_size=num_queries, reduce='mean') # Mean of squared distances, shape: [num_queries]
        E_X_squared = D_avg ** 2                # Square of mean distance, shape: [num_queries]
        D_var = E_X2 - E_X_squared              # Distance variance, shape: [num_queries]
        D_var = torch.clamp(D_var, min=0.0)     # Ensure non-negative variance

        # Calculate neighbor centroid and offset
        nbr_centroid = scatter(nbr_coords, query_indices_per_neighbor, dim=0, dim_size=num_queries, reduce="mean")  # Shape: [num_queries, num_dims]
        Delta = nbr_centroid - query_pos  # Shape: [num_queries, num_dims]

        # Calculate covariance matrix
        nbr_coords_centered = nbr_coords - nbr_centroid[query_indices_per_neighbor]           # Centered neighbor coordinates, shape: [NumEdges, num_dims]
        cov_components = nbr_coords_centered.unsqueeze(2) * nbr_coords_centered.unsqueeze(1)  # Covariance components, shape: [NumEdges, num_dims, num_dims]
        cov_sum = scatter(cov_components, query_indices_per_neighbor, dim=0, dim_size=num_queries, reduce="sum")  # Covariance sum, shape: [num_queries, num_dims, num_dims]
        N_i_clamped = N_i.clone()
        N_i_clamped[N_i_clamped == 0] = 1.0                # Avoid division by zero
        cov_matrix = cov_sum / N_i_clamped.view(-1, 1, 1)  # Covariance matrix, shape: [num_queries, num_dims, num_dims]

        # Calculate PCA features (eigenvalues of covariance matrix)
        PCA_features = torch.zeros(num_queries, num_dims, device=device)
        if has_neighbors.any():
            cov_matrix_valid = cov_matrix[has_neighbors]
            eps = 1e-6
            eye = torch.eye(num_dims, device=device, dtype=cov_matrix_valid.dtype)
            cov_matrix_reg = cov_matrix_valid + eps * eye.unsqueeze(0)
            try:
                eigenvalues = torch.linalg.eigvalsh(cov_matrix_reg) 
                eigenvalues = eigenvalues.flip(dims=[1]) 
                PCA_features[has_neighbors] = eigenvalues
            except Exception as e:
                default_eigenvals = torch.ones(num_dims, device=device) * eps
                PCA_features[has_neighbors] = default_eigenvals.unsqueeze(0).expand(has_neighbors.sum(), -1)
        
        # Combine all features
        N_i_tensor = N_i.unsqueeze(1)      # Shape: [num_queries, 1]
        D_avg_tensor = D_avg.unsqueeze(1)  # Shape: [num_queries, 1]
        D_var_tensor = D_var.unsqueeze(1)  # Shape: [num_queries, 1]
        geo_features = torch.cat([N_i_tensor, D_avg_tensor, D_var_tensor, Delta, PCA_features], dim=1)  # Shape: [num_queries, num_features]

        # Set features to zero for query points with no neighbors
        geo_features[~has_neighbors] = 0.0

        feature_mean = geo_features.mean(dim=0, keepdim=True)
        feature_std = geo_features.std(dim=0, keepdim=True)
        feature_std[feature_std < 1e-6] = 1.0 
        geo_features_normalized = (geo_features - feature_mean) / feature_std

        return geo_features_normalized
    
    def _compute_pointnet_features_pyg(self, source_pos, query_pos, edge_index):
        """Computes PointNet-style features using PyG edge_index."""
        num_query_nodes = query_pos.shape[0]
        device = query_pos.device


        geo_features = torch.zeros((num_query_nodes, self.output_dim), device=device, dtype=query_pos.dtype)

        if edge_index.numel() == 0: 
            print("Warning: GeoEmbed (PointNet) received no edges.")
            return geo_features

        query_idx = edge_index[1].long()
        source_idx = edge_index[0].long()

        has_neighbors_mask = torch.bincount(query_idx, minlength=num_query_nodes) > 0

        if not torch.any(has_neighbors_mask): 
             return geo_features


        nbr_coords = source_pos[source_idx]                       # [NumEdges, D]
        query_coords_per_edge = query_pos[query_idx]              # [NumEdges, D]

        nbr_coords_centered = nbr_coords - query_coords_per_edge  # [NumEdges, D]

        nbr_features = self.pointnet_mlp(nbr_coords_centered)  # [NumEdges, HiddenDim (e.g., 64)]

        if self.pooling == 'max':
            pooled_features = scatter(nbr_features, query_idx, dim=0, dim_size=num_query_nodes, reduce='max')
        elif self.pooling == 'mean':
            pooled_features = scatter(nbr_features, query_idx, dim=0, dim_size=num_query_nodes, reduce='mean')
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        pointnet_output = self.fc(pooled_features)             # [NumQueryNodes, OutputDim]
        geo_features[has_neighbors_mask] = pointnet_output[has_neighbors_mask]

        return geo_features