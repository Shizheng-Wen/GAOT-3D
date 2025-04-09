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
                 latent_tokens: tuple = (64, 64, 64)):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = magno_config.lifting_channels 
        self.patch_size = attn_config.patch_size
        self.D = latent_tokens[0]
        self.H = latent_tokens[1]
        self.W = latent_tokens[2]


        # Initialize encoder, processor, and decoder
        self.encoder = self.init_encoder(input_size, magno_config)
        self.processor = self.init_processor(self.node_latent_size, attn_config)
        self.decoder = self.init_decoder(output_size, magno_config)
    
    def init_encoder(self, input_size, magno_config):
        return GNOEncoder(
            in_channels = input_size,
            out_channels = self.node_latent_size,
            gno_config = magno_config
        )
    
    def init_processor(self, node_latent_size, config):
        # Initialize the Vision Transformer processor
        self.patch_linear = nn.Linear(self.patch_size * self.patch_size * self.patch_size * self.node_latent_size,
                                      self.patch_size * self.patch_size * self.patch_size * self.node_latent_size)
    
        self.positional_embedding_name = config.positional_embedding
        self.positions = self.get_patch_positions()
        if self.positional_embedding_name == 'absolute':
            pos_emb = self.compute_absolute_embeddings(self.positions, self.patch_size * self.patch_size * self.patch_size * self.node_latent_size)
            self.register_buffer('positional_embeddings', pos_emb)

        setattr(config.attn_config, 'D', self.D)
        setattr(config.attn_config, 'H', self.H)
        setattr(config.attn_config, 'W', self.W)

        return Transformer(
            input_size=self.node_latent_size * self.patch_size * self.patch_size * self.patch_size,
            output_size=self.node_latent_size * self.patch_size * self.patch_size * self.patch_size,
            config=config
        )

    def init_decoder(self, output_size, magno_config):
        # Initialize the GNO decoder
        return GNODecoder(
            in_channels=self.node_latent_size,
            out_channels=output_size,
            gno_config=magno_config
        )

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
            pos_emb = self.compute_absolute_embeddings(pos, P * P * P * self.node_latent_size)
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
            batch = batch,
            latent_tokens = token,
            regional_nodes = rndata
        )
        return decoded

    def forward(self,
                batch: Batch,
                tokens: Optional[torch.Tensor] = None,
                condition: Optional[float] = None
                ) -> torch.Tensor:
        """
        Forward pass for GIVI model.

        Parameters
        ----------
        batch: Batch
            The input batch containing the data
        tokens: Optional[torch.Tensor]
            ND Tensor of shape [batch_size, n_token_nodes, n_dim]
        condition: Optional[float]
            The condition of the model

        Returns
        -------
        torch.Tensor
            The output tensor of shape [batch_size, n_physical_nodes, output_size]
        """
        # Encode: Map physical nodes to regional nodes using MAGNO Encoder
        rndata = self.encode(
            batch         =   batch, 
            token         =   tokens)

        # Process: Apply Vision Transformer on the regional nodes
        rndata = self.process(
            rndata      =   rndata, 
            condition   =   condition)

        # Decode: Map regional nodes back to physical nodes using MAGNO Decoder
        output = self.decode(
            rndata      =   rndata, 
            batch       =   batch,
            token       =   self.latent_tokens)

        return output

    def get_patch_positions(self):
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

    def compute_absolute_embeddings(self, positions, embed_dim):
        """
        Compute RoPE embeddings for the given positions.
        """
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(positions.size(0), -1)
        return pos_emb
