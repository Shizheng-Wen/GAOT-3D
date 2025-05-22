# src/model/layers/magno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius as pyg_radius # Import radius function
from torch_geometric.utils import coalesce
try:
    from torch_cluster import knn
    TORCH_CLUSTER_AVAILABLE = True
except ImportError:
    TORCH_CLUSTER_AVAILABLE = False
    print("Warning: torch_cluster not found. KNN neighbor strategy will not work.")

import torch_geometric as pyg

from typing import Literal, Optional, Tuple
from dataclasses import dataclass, replace, field
from typing import Union, Tuple, Optional
from .geoembed import GeometricEmbedding
from .mlp import LinearChannelMLP, ChannelMLP
from .integral_transform import IntegralTransform
from ...utils.scale import rescale
############
# MAGNO Config
############
@dataclass
class MAGNOConfig:
    # GNO parameters
    use_gno: bool = True                            # Whether to use MAGNO
    gno_coord_dim: int = 2                          # Coordinate dimension
    gno_radius: float = 0.033                       # Radius for neighbor finding
    ## GNOEncoder
    lifting_channels: int = 16                      # Number of channels in the lifting MLP
    encoder_feature_attr: str = 'x'                 # Feature attribute name for the encoder
    in_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64, 64]) # Hidden layers in the GNO encoder MLP
    in_gno_transform_type: str = 'linear'           # Transformation type for the GNO encoder MLP
    ## GNODecoder
    projection_channels: int = 256                  # Number of channels in the projection MLP
    out_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64]) # Hidden layers in the GNO decoder MLP
    out_gno_transform_type: str = 'linear'          # Transformation type for the GNO decoder MLP
    # multiscale aggregation
    scales: list = field(default_factory=lambda: [1.0]) # Scales for multi-scale aggregation
    use_scale_weights: bool = False                     # Whether to use scale weights
    use_graph_cache: bool = True                        # Whether to use graph cache
    gno_use_torch_cluster: bool = False                 # Whether to use torch_cluster for neighbor finding
    gno_use_torch_scatter: str = True                   # Whether to use torch_scatter for neighbor finding
    node_embedding: bool = False                        # Whether to use node embedding
    use_attn: Optional[bool] = None                     # Whether to use attention
    attention_type: str = 'cosine'                      #  # Type of attention, supports ['cosine', 'dot_product']
    # Geometric embedding
    use_geoembed: bool = False                          # Whether to use geometric embedding
    embedding_method: str = 'statistical'               # Method for geometric embedding, supports ['statistical', 'pointnet']
    pooling: str = 'max'                                # Pooling method for pointnet geoembedding, supports ['max', 'mean']
    # Sampling
    sampling_strategy: Optional[str] = None            # Sampling strategy, supports ['max_neighbors', 'ratio']
    max_neighbors: Optional[int] = None                # Maximum number of neighbors
    sample_ratio: Optional[float] = None               # Sampling ratio
    # neighbor finding strategy
    neighbor_strategy: str = 'radius'                  # Neighbor finding strategy, supports ['radius', 'knn', 'bidirectional']
    # Dataset
    precompute_edges: bool = True                      # Flag for model to load vs compute edges. This aligns with the update_pt_files_with_edges in DatasetConfig


############
# Utils Functions
############
def get_neighbor_strategy(
    neighbor_strategy: str,
    phys_pos: torch.Tensor,
    batch_idx_phys: torch.Tensor,
    latent_tokens_pos: torch.Tensor,
    batch_idx_latent: torch.Tensor,
    radius: float):
    """
    Get the neighbor strategy based on the provided string.
    Args:
        neighbor_strategy (str): The neighbor strategy to use.
        phys_pos (Tensor): Physical positions.
        batch_idx_phys (Tensor): Batch indices for physical positions.
        latent_tokens_pos (Tensor): Latent token positions.
        batch_idx_latent (Tensor): Batch indices for latent token positions.
        radius (float): Radius for neighbor finding.
    Returns:
        edge_index (Tensor): Edge index for the neighbors.
    """
    device = phys_pos.device
    edge_index_fwd, edge_index_rev_swapped = None, None

    if neighbor_strategy in ['radius', 'bidirectional']:
        edge_index_fwd = pyg_radius(
            x=phys_pos,               # Source = physical
            y=latent_tokens_pos,      # Query = latent
            r=radius,
            batch_x=batch_idx_phys,
            batch_y=batch_idx_latent,
            max_num_neighbors=1000 
        )
    
    if neighbor_strategy in ['knn', 'bidirectional']:
        # For each physical point (y), find nearest latent token (x)
        edge_index_rev = knn(
            x=latent_tokens_pos,      # Data points to search within (latent)
            y=phys_pos,               # Query points (physical)
            k=1,
            batch_x=batch_idx_latent, # Batch index for data points
            batch_y=batch_idx_phys    # Batch index for query points
        ) # Returns [2, TotalPhysicalNodes] where row 0=phys_idx, row 1=latent_idx

        # Swap the order of the edge_index to match the forward direction
        edge_index_rev_swapped = edge_index_rev.flip(0) # Swap the order of the edges
        # edge_index_rev_swapped = edge_index_rev[[1, 0], :]

    # --- Combine the edge indices based on the strategy ---
    if neighbor_strategy == 'radius':
        return edge_index_fwd if edge_index_fwd is not None else torch.empty((2,0), dtype=torch.long, device=device)
    elif neighbor_strategy == 'knn':
        return edge_index_rev_swapped if edge_index_rev_swapped is not None else torch.empty((2,0), dtype=torch.long, device=device)
    elif neighbor_strategy == 'bidirectional':
        if edge_index_fwd is None: edge_index_fwd = torch.empty((2,0), dtype=torch.long, device=device)
        if edge_index_rev_swapped is None: edge_index_rev_swapped = torch.empty((2,0), dtype=torch.long, device=device)
        combined = torch.cat([edge_index_fwd, edge_index_rev_swapped], dim=1)
        # Coalesce removes duplicates and sorts
        return coalesce(combined)
    else:
            raise ValueError(f"Unknown neighbor strategy: {neighbor_strategy}")


############
# MAGNOEncoder
############
class GNOEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config: MAGNOConfig):
        super().__init__()
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.lifting_channels = gno_config.lifting_channels
        self.coord_dim = gno_config.gno_coord_dim
        self.feature_attr_name = gno_config.encoder_feature_attr
        self.precompute_edges = gno_config.precompute_edges

        # --- Store Neighbor Finding Strategy ---
        self.neighbor_strategy = gno_config.neighbor_strategy
        if self.neighbor_strategy in ['knn', 'bidirectional'] and not TORCH_CLUSTER_AVAILABLE:
            raise ImportError(f"torch_cluster is required for neighbor_strategy='{self.neighbor_strategy}'")
        # --- Init GNO Layer ---

        ## --- Calculate MLP input dimension ---
        self.use_gno = gno_config.use_gno
        if self.use_gno:
            in_kernel_in_dim = self.coord_dim * 2 
            if gno_config.in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
                in_kernel_in_dim += in_channels 
        
            in_gno_channel_mlp_hidden_layers = gno_config.in_gno_channel_mlp_hidden_layers.copy()
            in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
            in_gno_channel_mlp_hidden_layers.append(self.lifting_channels) # Kernel MLP output dim

            self.gno = IntegralTransform(
                channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
                transform_type=gno_config.in_gno_transform_type,
                # use_torch_scatter determined globally now
                use_attn=gno_config.use_attn,
                coord_dim=self.coord_dim, # Pass coord_dim if attn used
                attention_type=gno_config.attention_type,
                sampling_strategy=gno_config.sampling_strategy,
                max_neighbors=gno_config.max_neighbors,
                sample_ratio=gno_config.sample_ratio
            )

            ## --- Init Lifting MLP ---
            self.lifting = ChannelMLP(
                in_channels=in_channels,
                out_channels=self.lifting_channels, # Output matches GNO kernel output
                n_layers=1
            )
        else:
            self.gno = None
            self.lifting = None
            if in_channels > 0: 
                print("Warning: GNOEncoder has input_channels > 0 but use_gno=False. Input features (batch.x) will be ignored by the encoder path.")

        # --- Init GeoEmbed （optional) ---
        self.use_geoembed = gno_config.use_geoembed
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding( 
                input_dim=self.coord_dim,
                output_dim=self.lifting_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * self.lifting_channels,
                out_channels=self.lifting_channels,
                n_layers=1
            )
        
        # --- Init Scale Weighting (optional) ---
        self.use_scale_weights = gno_config.use_scale_weights
        if self.use_scale_weights:
            # Weighting based on latent token positions
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16), nn.ReLU(), nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(
        self, 
        batch: 'pyg.data.Batch', 
        latent_tokens_pos: torch.Tensor,
        latent_tokens_batch_idx: torch. Tensor
        ) -> torch.Tensor: 
        """
        Args:
            batch (Batch): PyG batch object (pos, x, batch for physical).
            latent_tokens_pos (Tensor): Latent token coordinates [TotalLatentNodes, D].
            latent_tokens_batch_idx (Tensor): Batch index for latent tokens [TotalLatentNodes].
        """
        phys_pos = batch.pos          # [TotalNodes_phys, D]
        batch_idx_phys = batch.batch # [TotalNodes_phys]
        device = phys_pos.device
        num_graphs = batch.num_graphs
        num_latent_tokens_per_graph = latent_tokens_pos.shape[0] // num_graphs # Calculate M

        phys_feat = getattr(batch, self.feature_attr_name, None)
        if phys_feat is None:
            if self.use_gno:
                raise AttributeError(f"GNOEncoder requires feature attribute '{self.feature_attr_name}' but it was not found in the batch.")

        # --- Multi-Scale GNO encoding ---
        encoded_scales = []
        for scale_idx, scale in enumerate(self.scales):
            scaled_radius = self.gno_radius * scale
            # Dynamic Bipartite Neighbor Search: physical (data) -> latent (query)
            # --- Get Edge Index and Optional Counts ---
            if self.precompute_edges:
                edge_index_attr = f'encoder_edge_index_s{scale_idx}'
                counts_attr = f'encoder_query_counts_s{scale_idx}'
                if not hasattr(batch, edge_index_attr):
                     raise AttributeError(f"Batch object missing pre-computed '{edge_index_attr}'")
                edge_index = getattr(batch, edge_index_attr).to(device)
                # Load optional counts for GeoEmbed
                neighbor_counts = getattr(batch, counts_attr, None)
                if neighbor_counts is not None:
                    neighbor_counts = neighbor_counts.to(device)
            else:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy = self.neighbor_strategy,
                    phys_pos = phys_pos,               # Source = physical
                    batch_idx_phys = batch_idx_phys,        # Batch indices for physical
                    latent_tokens_pos = latent_tokens_pos,      # Query = latent
                    batch_idx_latent = latent_tokens_batch_idx, # Batch indices for latent
                    radius = scaled_radius
                )
                neighbor_counts = None
            # --- Conditional GNO Path ---
            if self.use_gno:
                ## --- Lifting MLP ---
                ## ChannelMLP expects [B, C, N] or [C, N*B], using latter
                phys_feat_lifted = self.lifting(phys_feat.transpose(0, 1)).transpose(0, 1) # [TotalNodes_phys, C_lifted]
                encoded_gno = self.gno(
                    y_pos=phys_pos,           # Source coords (physical)
                    x_pos=latent_tokens_pos, # Query coords (latent)
                    edge_index=edge_index,      # Computed neighbors
                    f_y=phys_feat_lifted,     # Source features (lifted physical)
                    batch_y=batch_idx_phys,   # Pass batch indices if needed by GNO internals (e.g., batch norm)
                    batch_x=latent_tokens_batch_idx
                ) # Output shape: [TotalNodes_latent, C_lifted]
            else:
                encoded_gno = None
            # --- Conditional GeoEmbed Path ---
            if self.use_geoembed:
                geo_embedding = self.geoembed(
                    source_pos = phys_pos,             # Input geometry (physical)
                    query_pos = latent_tokens_pos, # Query points (latent)
                    edge_index = edge_index,           # Pass edge_index if needed by implementation
                    batch_source = batch_idx_phys,       # Pass batch info if needed
                    batch_query = latent_tokens_batch_idx,
                    neighbors_counts = neighbor_counts        # Optional neighbor counts for GeoEmbed
                ) # Output shape: [TotalNodes_latent, C_lifted]
            else:
                geo_embedding = None
            
            # --- Combine GNO and GeoEmbed ---
            if self.use_gno and self.use_geoembed:
                combined = torch.cat([encoded_gno, geo_embedding], dim=-1)
                encoded_unpatched = self.recovery(combined.permute(1,0)).permute(1,0) # Apply recovery MLP
            elif self.use_gno:
                encoded_unpatched = encoded_gno
            elif self.use_geoembed:
                encoded_unpatched = geo_embedding
            else:
                raise ValueError("GNO and GeoEmbed are both disabled. No encoding will be performed.")

            encoded_scales.append(encoded_unpatched) # List of [TotalNodes_latent, C_lifted]

        # --- Aggregate Scales ---
        if len(encoded_scales) == 1:
             encoded_data = encoded_scales[0]
        else:
             encoded_stack = torch.stack(encoded_scales, dim=0) # [num_scales, TotalNodes_latent, C_lifted]
             if self.use_scale_weights:
                  # Weights depend on latent token positions (apply per node)
                  scale_w = self.scale_weighting(latent_tokens_pos) # [TotalNodes_latent, num_scales]
                  scale_w = self.scale_weight_activation(scale_w)       # [TotalNodes_latent, num_scales]
                  # Reshape weights for broadcasting: [num_scales, TotalNodes_latent, 1]
                  weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                  encoded_data = (encoded_stack * weights_reshaped).sum(dim=0) # [TotalNodes_latent, C_lifted]
             else:
                  encoded_data = encoded_stack.sum(dim=0) # [TotalNodes_latent, C_lifted]

        # Output is the aggregated latent features for the entire batch
        # Shape: [TotalNodes_latent, C_lifted]
        # The subsequent Transformer expects [B, SeqLen, HiddenDim]
        # We need to reshape/process encoded_data before passing to Transformer
        # Reshape to [B, M, C_lifted]
        encoded_data = encoded_data.view(num_graphs, num_latent_tokens_per_graph, self.lifting_channels)

        return encoded_data

############
# MAGNODecoder
############
class GNODecoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config: MAGNOConfig):
        super().__init__()
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.coord_dim = gno_config.gno_coord_dim
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.use_geoembed = gno_config.use_geoembed 
        self.use_scale_weights = gno_config.use_scale_weights
        self.precompute_edges = gno_config.precompute_edges # store flag

        # --- Store Neighbor Strategy ---
        self.neighbor_strategy = gno_config.neighbor_strategy
        if self.neighbor_strategy in ['knn', 'bidirectional'] and not TORCH_CLUSTER_AVAILABLE:
            raise ImportError(f"torch_cluster is required for neighbor_strategy='{self.neighbor_strategy}'")
        # ---

        # --- Calculate MLP input dimension ---
        out_kernel_in_dim = self.coord_dim * 2 
        if gno_config.out_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
             out_kernel_in_dim += self.in_channels 

        # --- Init GNO Layer ---
        out_gno_channel_mlp_hidden_layers = gno_config.out_gno_channel_mlp_hidden_layers.copy()
        out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_channel_mlp_hidden_layers.append(self.in_channels) 

        self.gno = IntegralTransform(
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.out_gno_transform_type,
            # use_torch_scatter determined globally
            use_attn=gno_config.use_attn,
            coord_dim=self.coord_dim,
            attention_type=gno_config.attention_type,
            sampling_strategy=gno_config.sampling_strategy,
            max_neighbors=gno_config.max_neighbors,
            sample_ratio=gno_config.sample_ratio
        )

        # --- Init Projection ---
        self.projection = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=gno_config.projection_channels,
            n_layers=2,
            n_dim=1,
        )

        # --- Init GeoEmbed (Optional) ---
        if self.use_geoembed:
             # Geoembed input dim = coord_dim, output dim = in_channels (to match GNO output)
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=self.in_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                n_layers=1
            )


        # --- Init Scale Weighting (Optional) ---
        if self.use_scale_weights:
             # Weighting based on physical query positions
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16), nn.ReLU(), nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(self,
                rndata_flat: torch.Tensor,        # Flattened latent features [TotalLatent, C_in]
                phys_pos_query: torch.Tensor,     # Physical query coords [TotalQuery, D]
                batch_idx_phys_query: torch.Tensor,# Batch index for physical query [TotalQuery]
                latent_tokens_pos: torch.Tensor,  # Latent token coords (source) [TotalLatent, D]
                latent_tokens_batch_idx: torch.Tensor, # Batch index for latent source [TotalLatent]
                batch: 'pyg.data.Batch' = None    # Optional batch object for precomputed edges
               ) -> torch.Tensor: # Return shape [TotalQuery, C_out]
        """
        Args:
            rndata_flat (Tensor): Latent features (source) [TotalLatentNodes, C_in].
            phys_pos_query (Tensor): Physical/Query coordinates (dest) [TotalQueryNodes, D].
            batch_idx_phys_query (Tensor): Batch index for physical/query nodes [TotalQueryNodes].
            latent_tokens_pos (Tensor): Latent token coordinates (source) [TotalLatentNodes, D].
            latent_tokens_batch_idx (Tensor): Batch index for latent tokens (source) [TotalLatentNodes].
            batch (Batch): Optional PyG batch object for precomputed edges.
        """
        device = rndata_flat.device

        # --- Multi-Scale GNO decoding ---
        decoded_scales = []
        for scale_idx, scale in enumerate(self.scales):
            scaled_radius = self.gno_radius * scale
            # Dynamic Bipartite Neighbor Search: latent (data) -> physical (query)

            # --- Get Edge Index and Optional Counts ---
            if self.precompute_edges:
                edge_index_attr = f'decoder_edge_index_s{scale_idx}'
                counts_attr = f'decoder_query_counts_s{scale_idx}' # Note: query for decoder is physical
                if not hasattr(batch, edge_index_attr):
                     raise AttributeError(f"Batch object missing pre-computed '{edge_index_attr}'")
                edge_index = getattr(batch, edge_index_attr).to(device)
                # Load optional counts for GeoEmbed
                neighbor_counts = getattr(batch, counts_attr, None)
                if neighbor_counts is not None:
                    neighbor_counts = neighbor_counts.to(device)
            else:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy = self.neighbor_strategy,
                    phys_pos = latent_tokens_pos,             # Source = latent
                    batch_idx_phys = latent_tokens_batch_idx, # Batch indices for latent
                    latent_tokens_pos = phys_pos_query,       # Query = physical
                    batch_idx_latent = batch_idx_phys_query,  # Batch indices for physical
                    radius = scaled_radius
                )
                neighbor_counts = None

            # GNO Layer Call
            decoded_unpatched = self.gno(
                y_pos=latent_tokens_pos,     # Source coords (latent)
                x_pos=phys_pos_query,        # Query coords (physical)
                edge_index=edge_index,       # Computed neighbors
                f_y=rndata_flat,             # Source features (latent)
                batch_y=latent_tokens_batch_idx,
                batch_x=batch_idx_phys_query
            ) # Output shape: [TotalNodes_phys, C_in]

            # --- GeoEmbed (Optional) ---
            if self.use_geoembed:
                 # Geoembed needs latent_tokens_batched as input_geom, phys_pos as query points
                 geoembedding = self.geoembed(
                    source_pos = latent_tokens_pos,
                    query_pos = phys_pos_query,
                    edge_index = edge_index, # If needed
                    batch_source = latent_tokens_batch_idx,
                    batch_query = batch_idx_phys_query,
                    neighbors_counts = neighbor_counts # Optional neighbor counts for GeoEmbed
                 ) # Output shape: [TotalNodes_phys, C_in]
                 combined = torch.cat([decoded_unpatched, geoembedding], dim=-1)
                 # Apply recovery MLP 
                 decoded_unpatched = self.recovery(combined.permute(1,0)).permute(1,0) # Output: [TotalNodes_phys, C_in]

            decoded_scales.append(decoded_unpatched) # List of [TotalNodes_phys, C_in]

        # --- Aggregate Scales ---
        if len(decoded_scales) == 1:
             decoded_data = decoded_scales[0]
        else:
             decoded_stack = torch.stack(decoded_scales, dim=0) # [num_scales, TotalNodes_phys, C_in]
             if self.use_scale_weights:
                  # Weights depend on physical query positions
                  scale_w = self.scale_weighting(phys_pos_query) # [TotalNodes_phys, num_scales]
                  scale_w = self.scale_weight_activation(scale_w) # [TotalNodes_phys, num_scales]
                  # Reshape weights for broadcasting: [num_scales, TotalNodes_phys, 1]
                  weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                  decoded_data = (decoded_stack * weights_reshaped).sum(dim=0) # [TotalNodes_phys, C_in]
             else:
                  decoded_data = decoded_stack.sum(dim=0) # [TotalNodes_phys, C_in]

        # --- Final Projection ---
        # Input shape [TotalNodes_phys, C_in]
        decoded_data = decoded_data.permute(1,0)
        decoded_data = self.projection(decoded_data).permute(1,0) # Output shape [TotalNodes_phys, C_out] 
        return decoded_data