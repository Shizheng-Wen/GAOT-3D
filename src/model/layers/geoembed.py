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
        self.input_dim = input_dim # Dimension of coordinates (e.g., 3)
        self.output_dim = output_dim # Dimension of the embedding output
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
                # Removed final ReLU based on common practice for embeddings, add back if needed
            )
        elif self.method == 'pointnet':
            # PointNet MLPs process individual centered neighbor coords
            self.pointnet_mlp = nn.Sequential(
                nn.Linear(input_dim, 64), # Input is D-dim coord differences
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            # FC layer after pooling
            self.fc = nn.Sequential(
                nn.Linear(64, output_dim), # Input is pooled feature dim (64)
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forward(self,
                source_pos: torch.Tensor,
                query_pos: torch.Tensor,
                edge_index: torch.Tensor,
                batch_source: Optional[torch.Tensor] = None, # Included for completeness, but not used in current logic
                batch_query: Optional[torch.Tensor] = None,
                neighbors_counts: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Compute geometric embeddings using PyG batch format.

        Args:
            source_pos (Tensor): Coords of source nodes providing geometry [TotalSourceNodes, D].
            query_pos (Tensor): Coords of query nodes for which embeddings are computed [TotalQueryNodes, D].
            edge_index (Tensor): Bipartite edges [2, NumEdges], where edge_index[0] indexes query_pos,
                                 and edge_index[1] indexes source_pos.
            batch_source (Tensor, optional): Batch index for source nodes.
            batch_query (Tensor, optional): Batch index for query nodes.
            neighbors_counts (Tensor, optional): Number of neighbors for each query node.

        Returns:
            Tensor: Geometric embeddings for query nodes [TotalQueryNodes, output_dim].
        """
        if self.method == 'statistical':
            # Pass PyG inputs to the adapted statistical features function
            geo_features = self._compute_statistical_features_pyg(
                source_pos, query_pos, edge_index, neighbors_counts
            )
            # Normalize features (moved inside _compute_statistical_features_pyg)
            return self.mlp(geo_features) # Apply final MLP

        elif self.method == 'pointnet':
            # Pass PyG inputs to the adapted pointnet features function
            geo_features = self._compute_pointnet_features_pyg(
                source_pos, query_pos, edge_index
            )
            return geo_features # PointNet version includes the final FC layer

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
            edge_index (Tensor): Bipartite edges [2, NumEdges], where edge_index[0] indexes query_pos,
                                and edge_index[1] indexes source_pos.
            neighbors_counts (Tensor, optional): Number of neighbors for each query node.

        Returns:
            geo_features_normalized (torch.FloatTensor): The normalized geometric features, shape: [TotalQueryNodes, num_features]
        """
        # 基本参数
        num_queries = query_pos.shape[0]
        num_dims = query_pos.shape[1]
        device = query_pos.device
    
        # 从 edge_index 提取邻居信息
        neighbors_index = edge_index[1].long()  # 源节点的索引，形状: [NumEdges]
        query_indices_per_neighbor = edge_index[0].long()  # 每个邻居对应的查询节点索引，形状: [NumEdges]
        if neighbors_counts is None:
            num_neighbors_per_query = torch.bincount(query_indices_per_neighbor, minlength=num_queries).to(device)  # 每个查询节点的邻居数，形状: [num_queries]
        else:
            neighbors_counts = neighbors_counts.to(device)
            num_neighbors_per_query = neighbors_counts
        # 邻居数量和是否有邻居的标志
        N_i = num_neighbors_per_query.float()  # 形状: [num_queries]
        has_neighbors = N_i > 0  # 形状: [num_queries]

        # 获取邻居和查询节点的坐标
        nbr_coords = source_pos[neighbors_index]  # 形状: [NumEdges, num_dims]
        query_coords_per_neighbor = query_pos[query_indices_per_neighbor]  # 形状: [NumEdges, num_dims]

        # 计算距离统计特征
        distances = torch.norm(nbr_coords - query_coords_per_neighbor, dim=1)  # 形状: [NumEdges]
        D_avg = scatter(distances, query_indices_per_neighbor, dim=0, dim_size=num_queries, reduce="mean")  # 平均距离，形状: [num_queries]

        distances_squared = distances ** 2
        E_X2 = scatter(distances_squared, query_indices_per_neighbor, dim=0, dim_size=num_queries, reduce='mean') # 距离平方的均值，形状: [num_queries]
        E_X_squared = D_avg ** 2  # 平均距离的平方，形状: [num_queries]
        D_var = E_X2 - E_X_squared  # 距离方差，形状: [num_queries]
        D_var = torch.clamp(D_var, min=0.0)  # 确保方差非负

        # 计算邻居质心和偏移
        nbr_centroid = scatter(nbr_coords, query_indices_per_neighbor, dim=0, dim_size=num_queries, reduce="mean")  # 形状: [num_queries, num_dims]
        Delta = nbr_centroid - query_pos  # 形状: [num_queries, num_dims]

        # 计算协方差矩阵
        nbr_coords_centered = nbr_coords - nbr_centroid[query_indices_per_neighbor]  # 中心化的邻居坐标，形状: [NumEdges, num_dims]
        cov_components = nbr_coords_centered.unsqueeze(2) * nbr_coords_centered.unsqueeze(1)  # 协方差分量，形状: [NumEdges, num_dims, num_dims]
        cov_sum = scatter(cov_components, query_indices_per_neighbor, dim=0, dim_size=num_queries, reduce="sum")  # 协方差和，形状: [num_queries, num_dims, num_dims]
        N_i_clamped = N_i.clone()
        N_i_clamped[N_i_clamped == 0] = 1.0  # 避免除以零
        cov_matrix = cov_sum / N_i_clamped.view(-1, 1, 1)  # 协方差矩阵，形状: [num_queries, num_dims, num_dims]

        # 计算 PCA 特征（协方差矩阵的特征值）
        PCA_features = torch.zeros(num_queries, num_dims, device=device)
        if has_neighbors.any():
            cov_matrix_valid = cov_matrix[has_neighbors]
            eigenvalues = torch.linalg.eigvalsh(cov_matrix_valid)  # 计算特征值
            eigenvalues = eigenvalues.flip(dims=[1])  # 按降序排列
            PCA_features[has_neighbors] = eigenvalues
        
        # 组合所有特征
        N_i_tensor = N_i.unsqueeze(1)  # 形状: [num_queries, 1]
        D_avg_tensor = D_avg.unsqueeze(1)  # 形状: [num_queries, 1]
        D_var_tensor = D_var.unsqueeze(1)  # 形状: [num_queries, 1]
        geo_features = torch.cat([N_i_tensor, D_avg_tensor, D_var_tensor, Delta, PCA_features], dim=1)  # 形状: [num_queries, num_features]

        # 对于没有邻居的查询点，特征置零
        geo_features[~has_neighbors] = 0.0

        # 特征标准化
        feature_mean = geo_features.mean(dim=0, keepdim=True)
        feature_std = geo_features.std(dim=0, keepdim=True)
        feature_std[feature_std < 1e-6] = 1.0  # 防止除以接近零的标准差
        geo_features_normalized = (geo_features - feature_mean) / feature_std

        return geo_features_normalized
    
    def _compute_pointnet_features_pyg(self, source_pos, query_pos, edge_index):
        """Computes PointNet-style features using PyG edge_index."""
        num_query_nodes = query_pos.shape[0]
        device = query_pos.device

        # Initialize output tensor
        geo_features = torch.zeros((num_query_nodes, self.output_dim), device=device, dtype=query_pos.dtype)

        if edge_index.numel() == 0: # Handle case with no edges
            print("Warning: GeoEmbed (PointNet) received no edges.")
            return geo_features

        query_idx = edge_index[0]
        source_idx = edge_index[1]

        # Check which query nodes actually have neighbors
        has_neighbors_mask = torch.bincount(query_idx, minlength=num_query_nodes) > 0

        if not torch.any(has_neighbors_mask): # If no node has neighbors
             return geo_features

        # Get coordinates involved in edges
        nbr_coords = source_pos[source_idx]  # [NumEdges, D]
        query_coords_per_edge = query_pos[query_idx] # [NumEdges, D]

        # Center neighbor coordinates relative to their query point
        nbr_coords_centered = nbr_coords - query_coords_per_edge  # [NumEdges, D]

        # Apply PointNet MLP to centered coordinates
        nbr_features = self.pointnet_mlp(nbr_coords_centered)  # [NumEdges, HiddenDim (e.g., 64)]

        # Apply pooling per query node
        if self.pooling == 'max':
            pooled_features = scatter(nbr_features, query_idx, dim=0, dim_size=num_query_nodes, reduce='max')
            if HAS_TORCH_SCATTER: pooled_features = pooled_features[0] # Discard argmax
        elif self.pooling == 'mean':
            pooled_features = scatter(nbr_features, query_idx, dim=0, dim_size=num_query_nodes, reduce='mean')
        else:
            # Should have been caught in __init__
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        # Apply final fully connected layer to pooled features
        # Note: pooled_features shape is [NumQueryNodes, HiddenDim]
        pointnet_output = self.fc(pooled_features) # [NumQueryNodes, OutputDim]

        # Place computed features into the output tensor, only for nodes that had neighbors
        geo_features[has_neighbors_mask] = pointnet_output[has_neighbors_mask]

        return geo_features