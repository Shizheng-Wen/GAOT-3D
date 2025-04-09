# src/model/layers/magno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius as pyg_radius # Import radius function
import torch_geometric as pyg

from typing import Literal, Optional, Tuple

from dataclasses import dataclass, replace, field
from typing import Union, Tuple, Optional
from .utils.geoembed import GeometricEmbedding
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
            sampling_strategy=gno_config.gno_neighbor_sampling_strategy,
            max_neighbors=gno_config.gno_max_neighbors,
            sample_ratio=gno_config.gno_neighbor_sample_ratio
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


    def forward(self, batch: 'pyg.data.Batch', latent_tokens: torch.Tensor):
        """
        Args:
            batch (Batch): PyG batch object containing pos (physical coords), x (physical features), batch (indices).
            latent_tokens (Tensor): Latent token coordinates [M, D], assumed batch-independent.
        """
        phys_pos = batch.pos          # [TotalNodes_phys, D]
        phys_feat = batch.x          # [TotalNodes_phys, C_in]
        batch_idx_phys = batch.batch # [TotalNodes_phys]
        device = phys_pos.device
        num_graphs = batch.num_graphs

        # --- Lifting ---
        # Apply lifting to physical features
        # ChannelMLP expects [B, C, N] or [C, N*B], using latter
        phys_feat_lifted = self.lifting(phys_feat.transpose(0, 1)).transpose(0, 1) # [TotalNodes_phys, C_lifted]

        # --- Prepare Latent Tokens for Batch ---
        num_latent_tokens = latent_tokens.shape[0]
        latent_tokens_dev = latent_tokens.to(device)
        # Create batch index for latent tokens (repeat M times for each graph)
        batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(num_latent_tokens)
        # Repeat latent token coords for each graph in the batch
        latent_tokens_batched = latent_tokens_dev.repeat(num_graphs, 1) # [TotalNodes_latent, D]

        # --- Multi-Scale GNO encoding ---
        encoded_scales = []
        for scale in self.scales:
            scaled_radius = self.gno_radius * scale
            # Dynamic Bipartite Neighbor Search: physical (data) -> latent (query)
            edge_index = pyg_radius(
                x=phys_pos,           # Data points (source)
                y=latent_tokens_batched, # Query points (destination)
                r=scaled_radius,
                batch_x=batch_idx_phys,  # Batch indices for phys_pos
                batch_y=batch_idx_latent, # Batch indices for latent_tokens_batched
                #max_num_neighbors=int(num_latent_tokens*phys_pos.shape[0]*0.2) # Heuristic limit, adjust
            ) # Returns edge_index [2, NumEdges] where edge_index[0] indexes latent, edge_index[1] indexes physical

            # GNO Layer Call
            encoded_unpatched = self.gno(
                y_pos=phys_pos,           # Source coords (physical)
                x_pos=latent_tokens_batched, # Query coords (latent)
                edge_index=edge_index,      # Computed neighbors
                f_y=phys_feat_lifted,     # Source features (lifted physical)
                batch_y=batch_idx_phys,   # Pass batch indices if needed by GNO internals (e.g., batch norm)
                batch_x=batch_idx_latent
            ) # Output shape: [TotalNodes_latent, C_lifted]

            # --- GeoEmbed (Optional) ---
            if self.use_geoembed:
                 # Geoembed needs careful adaptation for bipartite edge_index
                 # Pass relevant info: phys_pos, latent_tokens_batched, edge_index, batch info?
                 # Assuming geoembed internal logic is updated or doesn't need edge_index directly
                 # Placeholder: Pass coordinates and let geoembed handle indexing/aggregation
                 geoembedding = self.geoembed(
                     phys_pos,             # Input geometry (physical)
                     latent_tokens_batched, # Query points (latent)
                     edge_index,           # Pass edge_index if needed by implementation
                     batch_idx_phys,       # Pass batch info if needed
                     batch_idx_latent
                 ) # Output shape: [TotalNodes_latent, C_lifted]

                 combined = torch.cat([encoded_unpatched, geoembedding], dim=-1)
                 # Recovery MLP expects [N, C] -> Use Linear instead of ChannelMLP? Or transpose?
                 # Using LinearChannelMLP adapted for node features:
                 recovery_mlp = LinearChannelMLP(layers=[2 * self.lifting_channels, self.lifting_channels], n_layers=1)
                 encoded_unpatched = recovery_mlp(combined.to(device)) # Apply recovery MLP


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
        encoded_data = encoded_data.view(num_graphs, num_latent_tokens, self.lifting_channels)

        return encoded_data


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
            sampling_strategy=gno_config.gno_neighbor_sampling_strategy,
            max_neighbors=gno_config.gno_max_neighbors,
            sample_ratio=gno_config.gno_neighbor_sample_ratio
        )

        # --- Init Projection ---
        # Use LinearChannelMLP or standard Linear for node features
        self.projection = LinearChannelMLP(
             layers=[self.in_channels, gno_config.projection_channels, self.out_channels],
             n_layers=2 # Example: 2 layers
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
             self.recovery = LinearChannelMLP(
                    layers=[2 * self.in_channels, gno_config.projection_channels, self.in_channels],
             ) # Adapt layers


        # --- Init Scale Weighting (Optional) ---
        self.use_scale_weights = gno_config.use_scale_weights
        if self.use_scale_weights:
             # Weighting based on physical query positions
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16), nn.ReLU(), nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)


    def forward(self, rndata_batched: torch.Tensor, batch: 'pyg.data.Batch', latent_tokens: torch.Tensor):
        """
        Args:
            rndata_batched (Tensor): Latent features from processor [B, M, C_in].
            batch (Batch): PyG batch object containing pos (physical coords), batch (indices).
            latent_tokens (Tensor): Latent token coordinates [M, D].
        """
        phys_pos = batch.pos          # [TotalNodes_phys, D]
        batch_idx_phys = batch.batch # [TotalNodes_phys]
        device = phys_pos.device
        num_graphs = batch.num_graphs
        num_phys_nodes_total = phys_pos.shape[0]

        # --- Prepare Latent Tokens & Features for Batch ---
        num_latent_tokens = latent_tokens.shape[0]
        latent_tokens_dev = latent_tokens.to(device)
        # Repeat latent token coords for batch
        latent_tokens_batched = latent_tokens_dev.repeat(num_graphs, 1) # [TotalNodes_latent, D]
        batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(num_latent_tokens)

        # Reshape processor output rndata_batched [B, M, C_in] -> [TotalNodes_latent, C_in]
        rndata_flat = rndata_batched.view(-1, self.in_channels) # [TotalNodes_latent, C_in]

        # --- Multi-Scale GNO decoding ---
        decoded_scales = []
        for scale in self.scales:
            scaled_radius = self.gno_radius * scale
            # Dynamic Bipartite Neighbor Search: latent (data) -> physical (query)
            edge_index = pyg_radius(
                x=latent_tokens_batched, # Data points (source) = latent
                y=phys_pos,              # Query points (destination) = physical
                r=scaled_radius,
                batch_x=batch_idx_latent, # Batch indices for latent
                batch_y=batch_idx_phys,   # Batch indices for physical
                max_num_neighbors=int(num_latent_tokens*phys_pos.shape[0]*0.2) # Heuristic
            ) # edge_index[0] indexes physical, edge_index[1] indexes latent

            # GNO Layer Call
            decoded_unpatched = self.gno(
                y_pos=latent_tokens_batched, # Source coords (latent)
                x_pos=phys_pos,              # Query coords (physical)
                edge_index=edge_index,       # Computed neighbors
                f_y=rndata_flat,             # Source features (latent)
                batch_y=batch_idx_latent,
                batch_x=batch_idx_phys
            ) # Output shape: [TotalNodes_phys, C_in]

            # --- GeoEmbed (Optional) ---
            if self.use_geoembed:
                 # Geoembed needs latent_tokens_batched as input_geom, phys_pos as query points
                 geoembedding = self.geoembed(
                      latent_tokens_batched,
                      phys_pos,
                      edge_index, # If needed
                      batch_idx_latent,
                      batch_idx_phys
                 ) # Output shape: [TotalNodes_phys, C_in]
                 combined = torch.cat([decoded_unpatched, geoembedding], dim=-1)
                 # Apply recovery MLP
                 decoded_unpatched = self.recovery(combined) # Output: [TotalNodes_phys, C_in]


            decoded_scales.append(decoded_unpatched) # List of [TotalNodes_phys, C_in]

        # --- Aggregate Scales ---
        if len(decoded_scales) == 1:
             decoded_data = decoded_scales[0]
        else:
             decoded_stack = torch.stack(decoded_scales, dim=0) # [num_scales, TotalNodes_phys, C_in]
             if self.use_scale_weights:
                  # Weights depend on physical query positions
                  scale_w = self.scale_weighting(phys_pos) # [TotalNodes_phys, num_scales]
                  scale_w = self.scale_weight_activation(scale_w) # [TotalNodes_phys, num_scales]
                  # Reshape weights for broadcasting: [num_scales, TotalNodes_phys, 1]
                  weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                  decoded_data = (decoded_stack * weights_reshaped).sum(dim=0) # [TotalNodes_phys, C_in]
             else:
                  decoded_data = decoded_stack.sum(dim=0) # [TotalNodes_phys, C_in]

        # --- Final Projection ---
        # Input shape [TotalNodes_phys, C_in]
        output = self.projection(decoded_data) # Output shape [TotalNodes_phys, C_out]

        return output