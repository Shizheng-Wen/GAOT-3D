# src/model/layers/magno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius as pyg_radius # Import radius function
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
    gno_coord_dim: int = 2
    projection_channels: int = 256
    in_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64, 64])
    out_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64])
    lifting_channels: int = 16
    gno_radius: float = 0.033
    # multiscale aggregation
    scales: list = field(default_factory=lambda: [1.0])
    use_scale_weights: bool = False
    use_graph_cache: bool = True
    gno_use_torch_cluster: bool = False
    in_gno_transform_type: str = 'linear'
    out_gno_transform_type: str = 'linear'
    gno_use_torch_scatter: str = True
    node_embedding: bool = False
    use_attn: Optional[bool] = None 
    attention_type: str = 'cosine'
    use_geoembed: bool = False
    embedding_method: str = 'statistical'
    pooling: str = 'max'
    # Sampling
    sampling_strategy: Optional[str] = None
    max_neighbors: Optional[int] = None
    sample_ratio: Optional[float] = None

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

        # --- Calculate MLP input dimension ---
        in_kernel_in_dim = self.coord_dim * 2 
        if gno_config.in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
             in_kernel_in_dim += in_channels 

        # --- Init GNO Layer ---
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

        # --- Init Lifting MLP ---
        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=self.lifting_channels, # Output matches GNO kernel output
            n_layers=1
        )

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
        phys_feat = batch.pos          # [TotalNodes_phys, C_in]
        batch_idx_phys = batch.batch # [TotalNodes_phys]
        device = phys_pos.device
        num_graphs = batch.num_graphs
        num_latent_tokens_per_graph = latent_tokens_pos.shape[0] // num_graphs # Calculate M

        # --- Lifting ---
        # Apply lifting to physical features
        # ChannelMLP expects [B, C, N] or [C, N*B], using latter
        phys_feat_lifted = self.lifting(phys_feat.transpose(0, 1)).transpose(0, 1) # [TotalNodes_phys, C_lifted]

        # --- Multi-Scale GNO encoding ---
        encoded_scales = []
        for scale in self.scales:
            scaled_radius = self.gno_radius * scale
            # Dynamic Bipartite Neighbor Search: physical (data) -> latent (query)
            edge_index = pyg_radius(
                x=phys_pos,           # Data points (source)
                y=latent_tokens_pos, # Query points (destination)
                r=scaled_radius,
                batch_x=batch_idx_phys,  # Batch indices for phys_pos
                batch_y=latent_tokens_batch_idx, # Batch indices for latent_tokens_batched
                #max_num_neighbors=int(num_latent_tokens*phys_pos.shape[0]*0.2) # Heuristic limit, adjust
            ) # Returns edge_index [2, NumEdges] where edge_index[0] indexes latent, edge_index[1] indexes physical

            # GNO Layer Call
            encoded_unpatched = self.gno(
                y_pos=phys_pos,           # Source coords (physical)
                x_pos=latent_tokens_pos, # Query coords (latent)
                edge_index=edge_index,      # Computed neighbors
                f_y=phys_feat_lifted,     # Source features (lifted physical)
                batch_y=batch_idx_phys,   # Pass batch indices if needed by GNO internals (e.g., batch norm)
                batch_x=latent_tokens_batch_idx
            ) # Output shape: [TotalNodes_latent, C_lifted]

            # --- GeoEmbed (Optional) ---
            if self.use_geoembed:
                # Geoembed needs careful adaptation for bipartite edge_index
                # Pass relevant info: phys_pos, latent_tokens_batched, edge_index, batch info?
                # Assuming geoembed internal logic is updated or doesn't need edge_index directly
                # Placeholder: Pass coordinates and let geoembed handle indexing/aggregation
                geoembedding = self.geoembed(
                    phys_pos,             # Input geometry (physical)
                    latent_tokens_pos, # Query points (latent)
                    edge_index,           # Pass edge_index if needed by implementation
                    batch_idx_phys,       # Pass batch info if needed
                    latent_tokens_batch_idx
                ) # Output shape: [TotalNodes_latent, C_lifted]

                combined = torch.cat([encoded_unpatched, geoembedding], dim=-1)
                # Recovery MLP expects [N, C] -> Use Linear instead of ChannelMLP? Or transpose?
                # Using LinearChannelMLP adapted for node features:
                encoded_unpatched = self.recovery(combined.permute(1,0)).permute(1,0) # Apply recovery MLP


            encoded_scales.append(encoded_unpatched) # List of [TotalNodes_latent, C_lifted]

        # --- Aggregate Scales ---
        if len(encoded_scales) == 1:
             encoded_data = encoded_scales[0]
        else:
             encoded_stack = torch.stack(encoded_scales, dim=0) # [num_scales, TotalNodes_latent, C_lifted]
             if self.use_scale_weights:
                  # Weights depend on latent token positions (apply per node)
                  scale_w = self.scale_weighting(latent_tokens_batched) # [TotalNodes_latent, num_scales]
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
        self.use_geoembed = gno_config.use_geoembed
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
        self.use_scale_weights = gno_config.use_scale_weights
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
                latent_tokens_batch_idx: torch.Tensor # Batch index for latent source [TotalLatent]
               ) -> torch.Tensor: # Return shape [TotalQuery, C_out]
        """
        Args:
            rndata_flat (Tensor): Latent features (source) [TotalLatentNodes, C_in].
            phys_pos_query (Tensor): Physical/Query coordinates (dest) [TotalQueryNodes, D].
            batch_idx_phys_query (Tensor): Batch index for physical/query nodes [TotalQueryNodes].
            latent_tokens_pos (Tensor): Latent token coordinates (source) [TotalLatentNodes, D].
            latent_tokens_batch_idx (Tensor): Batch index for latent tokens (source) [TotalLatentNodes].
        """
        device = rndata_flat.device

        # --- Multi-Scale GNO decoding ---
        decoded_scales = []
        for scale in self.scales:
            scaled_radius = self.gno_radius * scale
            # Dynamic Bipartite Neighbor Search: latent (data) -> physical (query)
            
            edge_index = pyg_radius(
                x=latent_tokens_pos, # Data points (source) = latent
                y=phys_pos_query,              # Query points (destination) = physical
                r=scaled_radius,
                batch_x=latent_tokens_batch_idx, # Batch indices for latent
                batch_y=batch_idx_phys_query,   # Batch indices for physical
            ) # edge_index[0] indexes physical, edge_index[1] indexes latent
            
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
                      latent_tokens_pos,
                      phys_pos_query,
                      edge_index, # If needed
                      latent_tokens_batch_idx,
                      batch_idx_phys_query
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