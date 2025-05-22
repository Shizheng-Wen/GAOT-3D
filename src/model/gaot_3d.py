import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field

from .layers.attn import Transformer, TransformerConfig
from .layers.magno import MAGNOConfig
from .layers.magno import GNOEncoder, GNODecoder

from torch_geometric.data import Batch


class GAOT3D(nn.Module):
    """
    Geometry Aware Operator Transformer: 
    Multiscale Attentional Graph Neural Operator + U Vision Transformer + Multiscale Attentional Graph Neural Operator
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 magno_config: MAGNOConfig = MAGNOConfig(),
                 attn_config: TransformerConfig = TransformerConfig(),
                 latent_tokens: tuple = (32, 32, 32),
                 norm_domin: list = [(-1, -1, -1), (1, 1, 1)]):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = magno_config.lifting_channels 
        self.patch_size = attn_config.patch_size
        self.D, self.H, self.W = latent_tokens
        self.num_latent_tokens = self.D * self.H * self.W
        self.coord_dim = magno_config.gno_coord_dim

        # --- Create and register latent_tokens ---
        x_min, y_min, z_min = norm_domin[0]
        x_max, y_max, z_max = norm_domin[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, self.D),
            torch.linspace(y_min, y_max, self.H),
            torch.linspace(z_min, z_max, self.W),
            indexing="ij"
        )

        internal_latent_tokens = torch.stack(meshgrid, dim=-1).reshape(-1, self.coord_dim)
        self.register_buffer('latent_tokens', internal_latent_tokens)
        # --- End latent_tokens creation ---
        # Initialize encoder, processor, and decoder
        self.encoder = self.init_encoder(input_size, self.node_latent_size, magno_config)
        self.processor = self.init_processor(self.node_latent_size, attn_config)
        self.decoder = self.init_decoder(self.node_latent_size, output_size, magno_config)
    
    def init_encoder(self, input_size, node_latent_size, magno_config):
        return GNOEncoder(
            in_channels = input_size,
            out_channels = node_latent_size,
            gno_config = magno_config
        )
    
    def init_processor(self, node_latent_size, config):
        # Initialize the Vision Transformer processor
        self.patch_linear = nn.Linear(self.patch_size * self.patch_size * self.patch_size * self.node_latent_size,
                                      self.patch_size * self.patch_size * self.patch_size * self.node_latent_size)
    
        self.positional_embedding_name = config.positional_embedding
        self.positions = self._get_patch_positions()

        setattr(config.attn_config, 'D', self.D)
        setattr(config.attn_config, 'H', self.H)
        setattr(config.attn_config, 'W', self.W)

        return Transformer(
            input_size=self.node_latent_size * self.patch_size * self.patch_size * self.patch_size,
            output_size=self.node_latent_size * self.patch_size * self.patch_size * self.patch_size,
            config=config
        )

    def init_decoder(self, node_latent_size, output_size, magno_config):
        # Initialize the GNO decoder
        return GNODecoder(
            in_channels=node_latent_size,
            out_channels=output_size,
            gno_config=magno_config
        )

    def _get_patch_positions(self):
        """
        Generate positional embeddings for the patches.
        """
        num_patches_D = self.D // self.patch_size
        num_patches_H = self.H // self.patch_size
        num_patches_W = self.W // self.patch_size
        positions = torch.stack(torch.meshgrid(
                torch.arange(num_patches_D, dtype=torch.float32),
                torch.arange(num_patches_H, dtype=torch.float32),
                torch.arange(num_patches_W, dtype=torch.float32),
                indexing='ij'
            ), dim=-1).reshape(-1, 3)

        return positions

    def _compute_absolute_embeddings(self, positions, embed_dim):
        """
        Compute absolute positional embeddings based on geometric coordinates.

        Args:
            positions (torch.Tensor): Tensor of shape [num_tokens, dims] representing the coordinates of each token.
            embed_dim (int): The desired embedding dimension. Assumed to be even for simplicity.

        Returns:
            torch.Tensor: Positional embedding tensor of shape [num_tokens, embed_dim].
        """
        # Extract the number of tokens and dimensions from the input tensor
        num_tokens, dims = positions.shape

        # Compute the frequencies, one for each pair of embedding dimensions
        # omega_k = 1 / 10000^(2k / embed_dim) for k = 0, 1, ..., embed_dim//2 - 1
        freq = 1 / 10000 ** (2 * torch.arange(0, embed_dim//2, dtype=torch.float32, device=positions.device) / embed_dim)

        # Expand dimensions for broadcasting: positions becomes [num_tokens, dims, 1]
        pos_expanded = positions[:, :, None]
        # freq becomes [1, 1, embed_dim//2]
        freq_expanded = freq[None, None, :]

        # Compute angles for all tokens, dimensions, and frequencies
        # angles[i, d, k] = positions[i, d] * freq[k]
        angles = pos_expanded * freq_expanded  # Shape: [num_tokens, dims, embed_dim//2]

        # Compute sine and cosine values for all angles
        sin_values = torch.sin(angles)  # Shape: [num_tokens, dims, embed_dim//2]
        cos_values = torch.cos(angles)  # Shape: [num_tokens, dims, embed_dim//2]

        # Sum the sine and cosine values across all dimensions
        sum_sin = torch.sum(sin_values, dim=1)  # Shape: [num_tokens, embed_dim//2]
        sum_cos = torch.sum(cos_values, dim=1)  # Shape: [num_tokens, embed_dim//2]

        # Initialize the positional embedding tensor
        PE = torch.zeros(num_tokens, embed_dim, device=positions.device)

        # Assign summed sine values to even indices and cosine values to odd indices
        PE[:, 0::2] = sum_sin  # Even positions
        PE[:, 1::2] = sum_cos  # Odd positions

        return PE

    def encode(self, batch: Batch, token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        batch: Batch
            The input batch containing the data
        token: Optional[torch.Tensor]
            ND Tensor of shape [batch_size, n_token_nodes, n_dim]
        Returns
        ------- 
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        # Apply GNO encoder
        encoded = self.encoder(
            batch = batch,
            latent_tokens = token
        )
        return encoded

    def process(self,
                rndata: Optional[torch.Tensor] = None,
                condition: Optional[float] = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        graph:Graph
            regional to regional graph, a homogeneous graph
        rndata:Optional[torch.Tensor]
            ND Tensor of shape [..., n_regional_nodes, node_latent_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        batch_size = rndata.shape[0]
        n_regional_nodes = rndata.shape[1]
        C = rndata.shape[2]
        D, H, W = self.D, self.H, self.W
        assert n_regional_nodes == D * H * W, \
            f"n_regional_nodes ({n_regional_nodes}) is not equal to H ({H}) * W ({W})"

        P = self.patch_size
        assert D % P ==0 and H % P == 0 and W % P == 0, f"Dimensions must be divisible by patch size"
        num_patches_D = D // P
        num_patches_H = H // P
        num_patches_W = W // P
        num_patches = num_patches_D * num_patches_H * num_patches_W
        # Reshape to patches
        rndata = rndata.view(batch_size, D, H, W, C)
        rndata = rndata.view(batch_size, num_patches_D, P, num_patches_H, P, num_patches_W, P, C)
        rndata = rndata.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # [batch, nD, nH, nW, P, P, P, C]
        rndata = rndata.view(batch_size, num_patches, P * P * P * C)
        
        # Apply Vision Transformer
        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)  # shape [num_patches, 3]
        if self.positional_embedding_name == 'absolute':
            pos_emb = self._compute_absolute_embeddings(pos, P * P * P * self.node_latent_size)
            rndata = rndata + pos_emb
            relative_positions = None
    
        elif self.positional_embedding_name == 'rope':
            relative_positions = pos

        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)

        # Reshape back to the original shape
        rndata = rndata.view(batch_size, num_patches_D, num_patches_H, num_patches_W, P, P, P, C)
        rndata = rndata.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        rndata = rndata.view(batch_size, D * H * W, C)

        return rndata

    def decode(self, rndata: Optional[torch.Tensor] = None,
                batch: Batch = None,
                token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        rndata: Optional[torch.Tensor]
            ND Tensor of shape [..., n_regional_nodes, node_latent_size]
        batch: Batch
            The input batch containing the data
        token: Optional[torch.Tensor]
            ND Tensor of shape [batch_size, n_token_nodes, n_dim]
        Returns
        -------
        torch.Tensor
            The output tensor of shape [batch_size, n_physical_nodes, output_size]
        """
        decoded = self.decoder(
            rndata_batched = rndata,
            batch = batch,
            latent_tokens = token
        )
        return decoded

    def forward(self,
                batch: Batch,
                tokens_pos: Optional[torch.Tensor] = None,
                tokens_batch_idx: Optional[torch.Tensor] = None,
                query_coord_pos: Optional[torch.Tensor] = None,
                query_coord_batch_idx: Optional[torch.Tensor] = None,
                condition: Optional[float] = None
                ) -> torch.Tensor:
        """
        Forward pass for GAOT3D model using PyG Batch.

        Args:
            batch (Batch): Input PyG Batch (contains batch.pos, batch.x, batch.batch).
            tokens_pos (Tensor, optional): External latent token coordinates [TotalLatentNodes, D].
                                           If None, uses internal self.latent_tokens.
            tokens_batch_idx (Tensor, optional): Batch index for external tokens [TotalLatentNodes].
                                                 Required if tokens_pos is provided.
            query_coord_pos (Tensor, optional): External query coordinates for decoder [TotalQueryNodes, D].
                                                 If None, uses batch.pos from input batch.
            query_coord_batch_idx (Tensor, optional): Batch index for external query coords [TotalQueryNodes].
                                                       Required if query_coord_pos is provided.
            condition (float, optional): Condition parameter for processor.

        Returns:
            Tensor: Output tensor on physical/query nodes [TotalQueryNodes, OutputChannels].
        """
        num_graphs = batch.num_graphs
        device = batch.pos.device
        
        # --- Determine Latent Tokens ---
        if tokens_pos is None:
            assert tokens_batch_idx is None, "tokens_batch_idx should be None if tokens_pos is None"
            latent_tokens_batched = self.latent_tokens.to(device).repeat(num_graphs, 1)
            batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(self.num_latent_tokens)
        else:
            if tokens_batch_idx is None:
                latent_tokens_batched = tokens_pos.to(device).repeat(num_graphs, 1)
                batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(self.num_latent_tokens)
            else:
                assert tokens_pos.shape[0] == tokens_batch_idx.shape[0], "tokens_pos and tokens_batch_idx must have same length"
                assert tokens_batch_idx.max() == num_graphs - 1, "tokens_batch_idx does not match batch size"
                latent_tokens_batched = tokens_pos.to(device)
                batch_idx_latent = tokens_batch_idx.to(device)
        
        # -- End Determine Latent Tokens ---

        # --- Determine Decoder Query Coordinates ---
        if query_coord_pos is None:
            phys_pos_query_eff = batch.pos # [TotalPhysicalNodes, D]
            batch_idx_phys_query_eff = batch.batch # [TotalPhysicalNodes]
        else:
            assert query_coord_batch_idx is not None, "query_coord_batch_idx is required if query_coord_pos is provided"
            assert query_coord_pos.shape[0] == query_coord_batch_idx.shape[0], "query_coord_pos and query_coord_batch_idx must have same length"
            assert query_coord_batch_idx.max() == num_graphs - 1, "query_coord_batch_idx does not match batch size"
            phys_pos_query_eff = query_coord_pos.to(device)
            batch_idx_phys_query_eff = query_coord_batch_idx.to(device)

        # Encode: Map physical nodes to regional nodes using MAGNO Encoder
        rndata = self.encoder(
             batch=batch, # Contains phys_pos, phys_feat, batch_idx_phys
             latent_tokens_pos=latent_tokens_batched,
             latent_tokens_batch_idx=batch_idx_latent
             ) # Output shape: [B, M, C_lifted]

        # Process: Apply Vision Transformer on the regional nodes
        # Input shape: [B, M, C_lifted]
        rndata_proc = self.process(
            rndata=rndata,
            condition=condition) # Output shape: [B, M, C_lifted] 

        # Flatten processor output for Decoder: [B, M, C_lifted] -> [TotalLatentNodes, C_lifted]
        rndata_proc_flat = rndata_proc.view(-1, self.node_latent_size) # Use self.node_latent_size

        # Decode: Map regional nodes back to physical nodes using MAGNO Decoder
        # Output shape: [TotalQueryNodes, OutputChannels]
        output = self.decoder(
            rndata_flat=rndata_proc_flat,
            phys_pos_query=phys_pos_query_eff,
            batch_idx_phys_query=batch_idx_phys_query_eff,
            latent_tokens_pos=latent_tokens_batched,
            latent_tokens_batch_idx=batch_idx_latent,
            batch = batch
            )

        return output
