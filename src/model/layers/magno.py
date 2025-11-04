# src/model/layers/magno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius as pyg_radius # Import radius function
from torch_geometric.nn import knn as pyg_knn
from torch_geometric.utils import coalesce, dropout_edge
import torch_geometric as pyg

from typing import Literal, Optional, Tuple
from dataclasses import dataclass, replace, field
from typing import Union, Tuple, Optional, List, Any
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
    ## MAGNOEncoder
    lifting_channels: int = 16                      # Number of channels in the lifting MLP
    encoder_feature_attr: Any = 'x'                 # Feature attribute name for the encoder, supports str or list of str
    in_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64, 64]) # Hidden layers in the GNO encoder MLP
    in_gno_transform_type: str = 'linear'           # Transformation type for the GNO encoder MLP
    ## MAGNODecoder
    projection_channels: int = 256                  # Number of channels in the projection MLP
    out_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64]) # Hidden layers in the GNO decoder MLP
    out_gno_transform_type: str = 'linear'          # Transformation type for the GNO decoder MLP
    # MLP type selection
    mlp_type: str = 'channel'                       # MLP type to use, supports ['channel', 'linear']. 
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
    use_geoembed: Any = field(default_factory=lambda: [True, True])  # Whether to use geometric embedding, supports:
                                                        # bool: same setting for both encoder and decoder
                                                        # List[bool]: [encoder_setting, decoder_setting]
    embedding_method: str = 'statistical'               # Method for geometric embedding, supports ['statistical', 'pointnet']
    pooling: str = 'max'                                # Pooling method for pointnet geoembedding, supports ['max', 'mean']
    # Sampling
    sampling_strategy: Optional[str] = None            # Sampling strategy, supports ['max_neighbors', 'ratio']
    max_neighbors: Optional[int] = None                # Maximum number of neighbors
    sample_ratio: Optional[float] = None               # Sampling ratio
    # neighbor finding strategy
    neighbor_strategy: Any = 'radius'  # Neighbor finding strategy, supports:
                                       # str: same strategy for both encoder and decoder
                                       # List[str]: [encoder_strategy, decoder_strategy]
                                       # Available strategies: ['radius', 'knn', 'bidirectional'] for encoder
                                       # Available strategies: ['radius', 'knn', 'bidirectional', 'reverse'] for decoder
    k_neighbors: int = 1               # Number of nearest neighbors for knn strategy
    # Dataset
    precompute_edges: bool = True                            # Flag for model to load vs compute edges. This aligns with the update_pt_files_with_edges in DatasetConfig
    asynchronous_graph_building: bool = False                # Flag for model to build graphs in the ddataloader. This aligns with the precompute_edges 


############
# Utils Functions
############
def parse_neighbor_strategy(neighbor_strategy: Union[str, List[str]]) -> Tuple[str, str]:
    """
    Parse neighbor_strategy into encoder and decoder strategies.
    
    Args:
        neighbor_strategy: Either a string (same for both) or [encoder_strategy, decoder_strategy]
        
    Returns:
        Tuple[str, str]: (encoder_strategy, decoder_strategy)
        
    Examples:
        parse_neighbor_strategy('radius') -> ('radius', 'radius')
        parse_neighbor_strategy(['knn', 'bidirectional']) -> ('knn', 'bidirectional')
    """
    if isinstance(neighbor_strategy, str):
        return neighbor_strategy, neighbor_strategy
    elif isinstance(neighbor_strategy, list) and len(neighbor_strategy) == 2:
        return neighbor_strategy[0], neighbor_strategy[1]
    else:
        raise ValueError(f"neighbor_strategy must be str or list of length 2, got {neighbor_strategy}")

def parse_geoembed_strategy(use_geoembed: Union[bool, List[bool]]) -> Tuple[bool, bool]:
    """
    Parse use_geoembed into encoder and decoder settings.
    
    Args:
        use_geoembed: Either a bool (same for both) or [encoder_setting, decoder_setting]
        
    Returns:
        Tuple[bool, bool]: (use_geoembed_encoder, use_geoembed_decoder)
        
    Examples:
        parse_geoembed_strategy(True) -> (True, True)
        parse_geoembed_strategy(False) -> (False, False)
        parse_geoembed_strategy([True, False]) -> (True, False)
        parse_geoembed_strategy([False, True]) -> (False, True)
    """
    if isinstance(use_geoembed, bool):
        return use_geoembed, use_geoembed
    elif isinstance(use_geoembed, list) and len(use_geoembed) == 2:
        return use_geoembed[0], use_geoembed[1]
    else:
        raise ValueError(f"use_geoembed must be bool or list of length 2, got {use_geoembed}")
    
def get_neighbor_strategy(
    neighbor_strategy: str,
    phys_pos: torch.Tensor,
    batch_idx_phys: torch.Tensor,
    latent_tokens_pos: torch.Tensor,
    batch_idx_latent: torch.Tensor,
    radius: float,
    k_neighbors: int = 1,
    is_decoder: bool = False):
    """
    Get the neighbor strategy based on the provided string.
    
    For ENCODER (is_decoder=False):
        - latent tokens are query points, fetching info from physical points
        - knn: each physical point connects to k nearest latent tokens (phys->latent)
        - radius: latent tokens as centers, physical points within radius connect (phys->latent)
        - bidirectional: merge knn and radius strategies
    
    For DECODER (is_decoder=True):
        - physical points are query points, fetching info from latent tokens
        - knn: each physical point connects to k nearest latent tokens (latent->phys)
        - radius: physical points as centers, latent tokens within radius connect (latent->phys)
        - bidirectional: merge knn and radius strategies
        - reverse: direct inverse of encoder graph
    
    Args:
        neighbor_strategy (str): Strategy to use ['knn', 'radius', 'bidirectional', 'reverse']
        phys_pos (Tensor): Physical positions [N_phys, 3]
        batch_idx_phys (Tensor): Batch indices for physical positions [N_phys]
        latent_tokens_pos (Tensor): Latent token positions [N_latent, 3]
        batch_idx_latent (Tensor): Batch indices for latent token positions [N_latent]
        radius (float): Radius for neighbor finding
        k_neighbors (int): Number of nearest neighbors for knn strategy
        is_decoder (bool): Whether this is for decoder graph construction
    Returns:
        edge_index (Tensor): Edge index [2, N_edges], format depends on encoder/decoder
    """
    
    if is_decoder:
        return _get_decoder_strategy(
            neighbor_strategy, phys_pos, batch_idx_phys, 
            latent_tokens_pos, batch_idx_latent, radius, k_neighbors
        )
    else:
        return _get_encoder_strategy(
            neighbor_strategy, phys_pos, batch_idx_phys,
            latent_tokens_pos, batch_idx_latent, radius, k_neighbors
        )

def _get_encoder_strategy(
    neighbor_strategy: str,
    phys_pos: torch.Tensor,
    batch_idx_phys: torch.Tensor,
    latent_tokens_pos: torch.Tensor,
    batch_idx_latent: torch.Tensor,
    radius: float,
    k_neighbors: int):
    """
    Encoder strategies: latent tokens as query points, fetching from physical points
    Edge direction: physical -> latent (info flows from physical to latent)
    """ 
    device = phys_pos.device
    edge_index_knn = None
    edge_index_radius = None
    
    if neighbor_strategy in ['knn', 'bidirectional']:
        # Each physical point connects to k nearest latent tokens
        edge_index_knn = pyg_knn(
            x=latent_tokens_pos,      
            y=phys_pos,               
            k=k_neighbors,            
            batch_x=batch_idx_latent, 
            batch_y=batch_idx_phys    
        ) # Returns [phys_idx, latent_idx] 
    
    if neighbor_strategy in ['radius', 'bidirectional']:
        # Latent tokens as centers, find physical points within radius
        edge_index_radius_raw = pyg_radius(
            x=phys_pos,               
            y=latent_tokens_pos,      
            r=radius,
            batch_x=batch_idx_phys,
            batch_y=batch_idx_latent,
            #max_num_neighbors=50000
        ) # Returns [latent_idx, phys_idx]
        edge_index_radius = edge_index_radius_raw.flip(0)
    # Combine strategies
    if neighbor_strategy == 'knn':
        return edge_index_knn if edge_index_knn is not None else torch.empty((2,0), dtype=torch.long, device=device)
    elif neighbor_strategy == 'radius':
        return edge_index_radius if edge_index_radius is not None else torch.empty((2,0), dtype=torch.long, device=device)
    elif neighbor_strategy == 'bidirectional':
        edges = []
        if edge_index_knn is not None:
            edges.append(edge_index_knn)
        if edge_index_radius is not None:
            edges.append(edge_index_radius)
        
        if len(edges) == 0:
            return torch.empty((2,0), dtype=torch.long, device=device)
        elif len(edges) == 1:
            return edges[0]
        else:
            combined = torch.cat(edges, dim=1)
            return coalesce(combined)  # Remove duplicates and sort
    else:
        raise ValueError(f"Unknown encoder strategy: {neighbor_strategy}")

def _get_decoder_strategy(
    neighbor_strategy: str,
    phys_pos: torch.Tensor,
    batch_idx_phys: torch.Tensor,
    latent_tokens_pos: torch.Tensor,
    batch_idx_latent: torch.Tensor,
    radius: float,
    k_neighbors: int):
    """
    Decoder strategies: physical points as query points, fetching from latent tokens
    Edge direction: latent -> physical (info flows from latent to physical)
    """
    device = phys_pos.device
    edge_index_knn = None
    edge_index_radius = None
    
    if neighbor_strategy in ['knn', 'bidirectional']:
        # Each physical point connects to k nearest latent tokens
        edge_index_knn_raw = pyg_knn(
            x=latent_tokens_pos,      
            y=phys_pos,               
            k=k_neighbors,            
            batch_x=batch_idx_latent, 
            batch_y=batch_idx_phys    
        ) # Returns [phys_idx, latent_idx]
        edge_index_knn = edge_index_knn_raw.flip(0) # [latent_idx, phys_idx]
    
    if neighbor_strategy in ['radius', 'bidirectional']:
        # Physical points as centers, find latent tokens within radius
        edge_index_radius_raw = pyg_radius(
            x=latent_tokens_pos,      
            y=phys_pos,               
            r=radius,
            batch_x=batch_idx_latent,
            batch_y=batch_idx_phys,
            #max_num_neighbors=50000
        ) # Returns [phys_idx, latent_idx]
        edge_index_radius = edge_index_radius_raw.flip(0)
    
    if neighbor_strategy == 'reverse':
        # Use the reverse of encoder graph
        # Get encoder graph and flip edge direction
        encoder_edges = _get_encoder_strategy(
            'bidirectional',  # Use bidirectional as default for reverse
            phys_pos, batch_idx_phys,
            latent_tokens_pos, batch_idx_latent,
            radius, k_neighbors
        )
        # Encoder gives [phys_idx, latent_idx], we want [latent_idx, phys_idx]
        return encoder_edges.flip(0)
    
    # Combine strategies
    if neighbor_strategy == 'knn':
        return edge_index_knn if edge_index_knn is not None else torch.empty((2,0), dtype=torch.long, device=device)
    elif neighbor_strategy == 'radius':
        return edge_index_radius if edge_index_radius is not None else torch.empty((2,0), dtype=torch.long, device=device)
    elif neighbor_strategy == 'bidirectional':
        edges = []
        if edge_index_knn is not None:
            edges.append(edge_index_knn)
        if edge_index_radius is not None:
            edges.append(edge_index_radius)
        
        if len(edges) == 0:
            return torch.empty((2,0), dtype=torch.long, device=device)
        elif len(edges) == 1:
            return edges[0]
        else:
            combined = torch.cat(edges, dim=1)
            return coalesce(combined)  # Remove duplicates and sort
    else:
        raise ValueError(f"Unknown decoder strategy: {neighbor_strategy}")

def apply_neighbor_sampling(
    edge_index: torch.Tensor,
    num_query_nodes: int,
    device: torch.device,
    sampling_strategy: Optional[str] = None,
    max_neighbors: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    training: bool = True
) -> torch.Tensor:
    """
    Applies neighbor sampling based on the configured strategy.
    
    Args:
        edge_index (Tensor): Edge index [2, num_edges]
        num_query_nodes (int): Number of query nodes
        device (torch.device): Device for tensor operations
        sampling_strategy (str, optional): Sampling strategy ['max_neighbors', 'ratio']
        max_neighbors (int, optional): Maximum number of neighbors per node
        sample_ratio (float, optional): Ratio of edges to keep
        training (bool): Whether model is in training mode
        
    Returns:
        Tensor: Sampled edge index [2, num_sampled_edges]
    """
    if sampling_strategy is None:
        return edge_index
        
    num_total_original_edges = edge_index.shape[1]

    if num_query_nodes == 0 or num_total_original_edges == 0:
        return edge_index 

    # --- Strategy 1: Max Neighbors Per Node ---
    if sampling_strategy == 'max_neighbors':
        if max_neighbors is None:
            raise ValueError("max_neighbors must be provided when using 'max_neighbors' sampling strategy")
            
        dest_nodes = edge_index[1] 
        counts = torch.bincount(dest_nodes, minlength=num_query_nodes)
        needs_sampling_mask = counts > max_neighbors

        if not torch.any(needs_sampling_mask):
            return edge_index 

        keep_mask = torch.ones(num_total_original_edges, dtype=torch.bool, device=device)
        queries_to_sample_idx = torch.where(needs_sampling_mask)[0]

        for i in queries_to_sample_idx:
            node_edge_mask = (dest_nodes == i)
            node_edge_indices = torch.where(node_edge_mask)[0]
            num_node_edges = len(node_edge_indices) 

            perm = torch.randperm(num_node_edges, device=device)[:max_neighbors]
            edges_to_keep_for_node = node_edge_indices[perm]
            # Update mask: keep only sampled edges for this node
            node_keep_mask = torch.zeros_like(node_edge_mask) 
            node_keep_mask[edges_to_keep_for_node] = True
            keep_mask[node_edge_mask] = node_keep_mask[node_edge_mask]

        sampled_edge_index = edge_index[:, keep_mask]
        return sampled_edge_index

    # --- Strategy 2: Global Ratio Sampling ---
    elif sampling_strategy == 'ratio':
        if sample_ratio is None:
            raise ValueError("sample_ratio must be provided when using 'ratio' sampling strategy")
            
        if sample_ratio >= 1.0:
             return edge_index
        p_drop = 1.0 - sample_ratio
        sampled_edge_index, _ = dropout_edge(edge_index, p=p_drop, force_undirected=False, training=training)
        return sampled_edge_index

    else:
         raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")


############
# MAGNOEncoder
############
class MAGNOEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config: MAGNOConfig):
        super().__init__()
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.lifting_channels = gno_config.lifting_channels
        self.coord_dim = gno_config.gno_coord_dim
        self.feature_attr_name = gno_config.encoder_feature_attr
        self.precompute_edges = gno_config.precompute_edges
        self.mlp_type = gno_config.mlp_type

        # --- Store Neighbor Finding Strategy ---
        self.encoder_strategy, self.decoder_strategy = parse_neighbor_strategy(gno_config.neighbor_strategy)
        self.k_neighbors = gno_config.k_neighbors
        
        # --- Store Sampling Strategy ---
        self.sampling_strategy = gno_config.sampling_strategy
        self.max_neighbors = gno_config.max_neighbors
        self.sample_ratio = gno_config.sample_ratio
        if self.sampling_strategy == 'max_neighbors':
            print("Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'.")
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
                attention_type=gno_config.attention_type
            )

            ## --- Init Lifting MLP ---
            if gno_config.mlp_type == 'linear':
                self.lifting = LinearChannelMLP(
                    layers=[in_channels, self.lifting_channels]
                )
            else:
                self.lifting = ChannelMLP(
                    in_channels=in_channels,
                    out_channels=self.lifting_channels, # Output matches GNO kernel output
                    n_layers=1
                )
        else:
            self.gno = None
            self.lifting = None
            if in_channels > 0: 
                print("Warning: MAGNOEncoder has input_channels > 0 but use_gno=False. Input features (batch.x) will be ignored by the encoder path.")

        # --- Init GeoEmbed （optional) ---
        use_geoembed_encoder, use_geoembed_decoder = parse_geoembed_strategy(gno_config.use_geoembed)
        self.use_geoembed = use_geoembed_encoder  # Use encoder-specific setting
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding( 
                input_dim=self.coord_dim,
                output_dim=self.lifting_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            if gno_config.mlp_type == 'linear':
                self.recovery = LinearChannelMLP(
                    layers=[2 * self.lifting_channels, self.lifting_channels]
                )
            else:
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
        batch_idx_phys = batch.batch  # [TotalNodes_phys]
        device = phys_pos.device
        num_graphs = batch.num_graphs
        num_latent_tokens_per_graph = latent_tokens_pos.shape[0] // num_graphs # Calculate M
        if isinstance(self.feature_attr_name, list):
            phys_feats = []
            for attr_name in self.feature_attr_name:
                feat = getattr(batch, attr_name, None)
                if feat is None:
                    if self.use_gno:
                        raise AttributeError(f"MAGNOEncoder requires feature attribute '{attr_name}' but it was not found in the batch.")
                else:
                    phys_feats.append(feat)
            phys_feat = torch.cat(phys_feats, dim=-1) if phys_feats else None
        else:
            phys_feat = getattr(batch, self.feature_attr_name, None)
            if phys_feat is None:
                if self.use_gno:
                    raise AttributeError(f"MAGNOEncoder requires feature attribute '{self.feature_attr_name}' but it was not found in the batch.")
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
                # neighbor_counts = getattr(batch, counts_attr, None)
                # if neighbor_counts is not None:
                #     neighbor_counts = neighbor_counts.to(device)
                neighbor_counts = None
            else:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy = self.encoder_strategy,  # Use encoder-specific strategy
                    phys_pos = phys_pos,                        # Source = physical
                    batch_idx_phys = batch_idx_phys,            # Batch indices for physical
                    latent_tokens_pos = latent_tokens_pos,      # Query = latent
                    batch_idx_latent = latent_tokens_batch_idx, # Batch indices for latent
                    radius = scaled_radius,
                    k_neighbors = self.k_neighbors,               
                    is_decoder = False                           # This is encoder
                )
                neighbor_counts = None
            # --- Apply Neighbor Sampling ---
            edge_index = apply_neighbor_sampling(
                edge_index=edge_index,
                num_query_nodes=latent_tokens_pos.shape[0],
                device=device,
                sampling_strategy=self.sampling_strategy,
                max_neighbors=self.max_neighbors,
                sample_ratio=self.sample_ratio,
                training=self.training
            )
            # --- Conditional GNO Path ---
            if self.use_gno:
                ## --- Lifting MLP ---
                if self.mlp_type == 'linear':
                    phys_feat_lifted = self.lifting(phys_feat) # [TotalNodes_phys, C_lifted]
                else:
                    phys_feat_lifted = self.lifting(phys_feat.transpose(0, 1)).transpose(0, 1) # [TotalNodes_phys, C_lifted]
                encoded_gno = self.gno(
                    y_pos=phys_pos,           # Source coords (physical)
                    x_pos=latent_tokens_pos,  # Query coords (latent)
                    edge_index=edge_index,    # Computed neighbors
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
                    query_pos = latent_tokens_pos,     # Query points (latent)
                    edge_index = edge_index,           # Pass edge_index if needed by implementation
                    batch_source = batch_idx_phys,     # Pass batch info if needed
                    batch_query = latent_tokens_batch_idx,
                    neighbors_counts = neighbor_counts   # Optional neighbor counts for GeoEmbed
                ) # Output shape: [TotalNodes_latent, C_lifted]
            else:
                geo_embedding = None
            
            # --- Combine GNO and GeoEmbed ---
            if self.use_gno and self.use_geoembed:
                combined = torch.cat([encoded_gno, geo_embedding], dim=-1)
                if self.mlp_type == 'linear':
                    encoded_unpatched = self.recovery(combined) 
                else:
                    encoded_unpatched = self.recovery(combined.permute(1,0)).permute(1,0) 
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
                  scale_w = self.scale_weighting(latent_tokens_pos)     # [TotalNodes_latent, num_scales]
                  scale_w = self.scale_weight_activation(scale_w)       # [TotalNodes_latent, num_scales]
                  weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                  encoded_data = (encoded_stack * weights_reshaped).sum(dim=0) # [TotalNodes_latent, C_lifted]
             else:
                  encoded_data = encoded_stack.sum(dim=0) # [TotalNodes_latent, C_lifted]
 
        encoded_data = encoded_data.view(num_graphs, num_latent_tokens_per_graph, self.lifting_channels) # [B, M, C_lifted]

        return encoded_data

############
# MAGNODecoder
############
class MAGNODecoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config: MAGNOConfig):
        super().__init__()
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.coord_dim = gno_config.gno_coord_dim
        self.in_channels = in_channels 
        self.out_channels = out_channels
        use_geoembed_encoder, use_geoembed_decoder = parse_geoembed_strategy(gno_config.use_geoembed)
        self.use_geoembed = use_geoembed_decoder  # Use decoder-specific setting
        self.use_scale_weights = gno_config.use_scale_weights
        self.precompute_edges = gno_config.precompute_edges # store flag
        self.mlp_type = gno_config.mlp_type

        # --- Store Neighbor Strategy ---
        self.encoder_strategy, self.decoder_strategy = parse_neighbor_strategy(gno_config.neighbor_strategy)
        self.k_neighbors = gno_config.k_neighbors
        
        # --- Store Sampling Strategy ---
        self.sampling_strategy = gno_config.sampling_strategy
        self.max_neighbors = gno_config.max_neighbors
        self.sample_ratio = gno_config.sample_ratio
        if self.sampling_strategy == 'max_neighbors':
            print("Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'.")
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
            use_attn=gno_config.use_attn,
            coord_dim=self.coord_dim,
            attention_type=gno_config.attention_type
        )

        # --- Init Projection ---
        if gno_config.mlp_type == 'linear':
            self.projection = LinearChannelMLP(
                layers=[in_channels, gno_config.projection_channels, out_channels]
            )
        else:
            self.projection = ChannelMLP(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=gno_config.projection_channels,
                n_layers=2,
                n_dim=1,
            )

        # --- Init GeoEmbed (Optional) ---
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=self.in_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            if gno_config.mlp_type == 'linear':
                self.recovery = LinearChannelMLP(
                    layers=[2 * in_channels, in_channels]
                )
            else:
                self.recovery = ChannelMLP(
                    in_channels=2 * in_channels,
                    out_channels=in_channels,
                    n_layers=1
                )


        # --- Init Scale Weighting (Optional) ---
        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16), nn.ReLU(), nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(self,
                rndata_flat: torch.Tensor,         
                phys_pos_query: torch.Tensor,      
                batch_idx_phys_query: torch.Tensor,
                latent_tokens_pos: torch.Tensor,   
                latent_tokens_batch_idx: torch.Tensor, 
                batch: 'pyg.data.Batch' = None     
               ) -> torch.Tensor:                  
        """
        Args:
            rndata_flat (Tensor): Latent features (source) [TotalLatentNodes, C_in].
            phys_pos_query (Tensor): Physical/Query coordinates (dest) [TotalQueryNodes, D].
            batch_idx_phys_query (Tensor): Batch index for physical/query nodes [TotalQueryNodes].
            latent_tokens_pos (Tensor): Latent token coordinates (source) [TotalLatentNodes, D].
            latent_tokens_batch_idx (Tensor): Batch index for latent tokens (source) [TotalLatentNodes].
            batch (Batch): Optional PyG batch object for precomputed edges. [TotalQueryNodes, C_out]
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
                # neighbor_counts = getattr(batch, counts_attr, None)
                # if neighbor_counts is not None:
                #     neighbor_counts = neighbor_counts.to(device)
                neighbor_counts = None
            else:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy = self.decoder_strategy,  # Use decoder-specific strategy
                    phys_pos = phys_pos_query,                  # Physical nodes (query points)
                    batch_idx_phys = batch_idx_phys_query,      # Batch indices for physical
                    latent_tokens_pos = latent_tokens_pos,      # Latent tokens (data points)
                    batch_idx_latent = latent_tokens_batch_idx, # Batch indices for latent
                    radius = scaled_radius,
                    k_neighbors = self.k_neighbors,           
                    is_decoder = True                         
                )
                neighbor_counts = None
                
            # --- Apply Neighbor Sampling ---
            edge_index = apply_neighbor_sampling(
                edge_index=edge_index,
                num_query_nodes=phys_pos_query.shape[0],
                device=device,
                sampling_strategy=self.sampling_strategy,
                max_neighbors=self.max_neighbors,
                sample_ratio=self.sample_ratio,
                training=self.training
            )

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
                    edge_index = edge_index, 
                    batch_source = latent_tokens_batch_idx,
                    batch_query = batch_idx_phys_query,
                    neighbors_counts = neighbor_counts # Optional neighbor counts for GeoEmbed
                ) # Output shape: [TotalNodes_phys, C_in]
                combined = torch.cat([decoded_unpatched, geoembedding], dim=-1)
                if self.mlp_type == 'linear':
                    decoded_unpatched = self.recovery(combined) # Output: [TotalNodes_phys, C_in]
                else:
                    decoded_unpatched = self.recovery(combined.permute(1,0)).permute(1,0) # Output: [TotalNodes_phys, C_in]

            decoded_scales.append(decoded_unpatched) # List of [TotalNodes_phys, C_in]

        # --- Aggregate Scales ---
        if len(decoded_scales) == 1:
             decoded_data = decoded_scales[0]
        else:
             decoded_stack = torch.stack(decoded_scales, dim=0) # [num_scales, TotalNodes_phys, C_in]
             if self.use_scale_weights:
                  scale_w = self.scale_weighting(phys_pos_query) # [TotalNodes_phys, num_scales]
                  scale_w = self.scale_weight_activation(scale_w) # [TotalNodes_phys, num_scales]
                  weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                  decoded_data = (decoded_stack * weights_reshaped).sum(dim=0) # [TotalNodes_phys, C_in]
             else:
                  decoded_data = decoded_stack.sum(dim=0) # [TotalNodes_phys, C_in]

        # --- Final Projection ---
        if self.mlp_type == 'linear':
            decoded_data = self.projection(decoded_data) # Output shape [TotalNodes_phys, C_out]
        else:
            decoded_data = decoded_data.permute(1,0)     # [C_in, TotalNodes_phys]
            decoded_data = self.projection(decoded_data).permute(1,0) # shape [TotalNodes_phys, C_out] 
        return decoded_data