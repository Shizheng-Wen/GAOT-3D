import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
import numpy as np

from dataclasses import dataclass, replace, field
from typing import Union, Tuple, Optional

from .utils.magno_utils import Activation, segment_csr, NeighborSearch
from .utils.geoembed import GeometricEmbedding
from .mlp import LinearChannelMLP, ChannelMLP

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
    gno_use_open3d: bool = False
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

############
# Integral Transform (GNO)
############

class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined

    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on\
        each x) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)
    
    f : Input function to integrate against given\
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    channel_mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f
    channel_mlp_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the LinearChannelMLP class
    channel_mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        LinearChannelMLP class. Only used if channel_mlp_layers is
        given and channel_mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    use_torch_scatter : bool, default 'True'
        Whether to use torch_scatter's implementation of 
        segment_csr or our native PyTorch version. torch_scatter 
        should be installed by default, but there are known versioning
        issues on some linux builds of CPU-only PyTorch. Try setting
        to False if you experience an error from torch_scatter.
    """

    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_torch_scatter=True,
        use_attn=None,
        coord_dim=None,
        attention_type='cosine'
    ):
        super().__init__()

        assert channel_mlp is not None or channel_mlp_layers is not None

        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter
        self.use_attn = use_attn
        self.attention_type = attention_type

        if self.transform_type not in ["linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"]:
            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp
        
        if self.use_attn:
            if coord_dim is None:
                raise ValueError("coord_dim must be specified when use_attn is True")
            self.coord_dim = coord_dim

            if self.attention_type == 'dot_product':
                attention_dim = 64 
                self.query_proj = nn.Linear(self.coord_dim, attention_dim)
                self.key_proj = nn.Linear(self.coord_dim, attention_dim)
                self.scaling_factor = 1.0 / (attention_dim ** 0.5)
            elif self.attention_type == 'cosine':
                pass
            else:
                raise ValueError(f"Invalid attention_type: {self.attention_type}. Must be 'cosine' or 'dot_product'.")


    """"
    

    Assumes x=y if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """

    def forward(self, y, neighbors, x=None, f_y=None, weights=None):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets A(x) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        if x is None:
            x = y
        
        rep_features = y[neighbors["neighbors_index"]]

        # batching only matters if f_y (latent embedding) values are provided
        batched = False
        # f_y has a batch dim IFF batched=True
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors["neighbors_index"], :]
            elif f_y.ndim == 2:
                batched = False
                in_features = f_y[neighbors["neighbors_index"]]

        num_reps = (
            neighbors["neighbors_row_splits"][1:]
            - neighbors["neighbors_row_splits"][:-1]
        )

        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        # attention usage
        if self.use_attn:
            query_coords = self_features[:, :self.coord_dim]
            key_coords = rep_features[:, :self.coord_dim]

            if self.attention_type == 'dot_product':
                query = self.query_proj(query_coords)  # [num_neighbors, attention_dim]
                key = self.key_proj(key_coords)        # [num_neighbors, attention_dim]

                attention_scores = torch.sum(query * key, dim=-1) * self.scaling_factor  # [num_neighbors] 
                
            elif self.attention_type == 'cosine':
                query_norm = F.normalize(query_coords, p=2, dim=-1)
                key_norm = F.normalize(key_coords, p=2, dim=-1)
                attention_scores = torch.sum(query_norm * key_norm, dim=-1)  # [num_neighbors]
            else:
                raise ValueError(f"Invalid attention_type: {self.attention_type}. Must be 'cosine' or 'dot_product'.")

            splits = neighbors["neighbors_row_splits"]
            attention_weights = self.segment_softmax(attention_scores, splits)
        else:
            attention_weights = None


        agg_features = torch.cat([rep_features, self_features], dim=-1)
        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                # repeat agg features for every example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        rep_features = self.channel_mlp(agg_features) # TODOz:这一步累计的计算图巨大[280468]
    
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features
        
        if self.use_attn:
            rep_features = rep_features * attention_weights.unsqueeze(-1)

        if weights is not None:
            assert weights.ndim == 1, "Weights must be of dimension 1 in all cases"
            nbr_weights = weights[neighbors["neighbors_index"]]
            # repeat weights along batch dim if batched
            if batched:
                nbr_weights = nbr_weights.repeat(
                    [batch_size] + [1] * nbr_weights.ndim
                )
            rep_features = nbr_weights * rep_features
            reduction = "sum"
        else:
            reduction = "mean" if not self.use_attn else "sum"

        splits = neighbors["neighbors_row_splits"]
        if batched:
            splits = splits.repeat([batch_size] + [1] * splits.ndim)

        out_features = segment_csr(rep_features, splits, reduce=reduction, use_scatter=self.use_torch_scatter)

        return out_features

    def segment_softmax(self, attention_scores, splits):
        """
        apply soft_max on every regional node neighbors.

        Parameters：
        - attention_scores: [num_neighbors]，attention scores
        - splits: neighbors split information

        Return：
        - attention_weights: [num_neighbors]，normalized attention scores
        """
        max_values = segment_csr(
            attention_scores, splits, reduce='max', use_scatter=self.use_torch_scatter
        )
        max_values_expanded = max_values.repeat_interleave(
            splits[1:] - splits[:-1], dim=0
        )
        attention_scores = attention_scores - max_values_expanded
        exp_scores = torch.exp(attention_scores)
        sum_exp = segment_csr(
            exp_scores, splits, reduce='sum', use_scatter=self.use_torch_scatter
        )
        sum_exp_expanded = sum_exp.repeat_interleave(
            splits[1:] - splits[:-1], dim=0
        )
        attention_weights = exp_scores / sum_exp_expanded
        return attention_weights



##########
# GNOEncoder
##########
class GNOEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)
        self.gno_radius = gno_config.gno_radius
        self.graph_cache = None 
        self.use_graph_cache = gno_config.use_graph_cache
        self.scales = gno_config.scales
        self.use_scale_weights = gno_config.use_scale_weights

        in_kernel_in_dim = gno_config.gno_coord_dim * 2
        coord_dim = gno_config.gno_coord_dim 
        if gno_config.node_embedding:
            in_kernel_in_dim = gno_config.gno_coord_dim * 4 * 2 * 2  # 32
            coord_dim = gno_config.gno_coord_dim * 4 * 2
        if gno_config.in_gno_transform_type == "nonlinear" or gno_config.in_gno_transform_type == "nonlinear_kernelonly":
            in_kernel_in_dim += in_channels

        in_gno_channel_mlp_hidden_layers = gno_config.in_gno_channel_mlp_hidden_layers.copy()
        in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
        in_gno_channel_mlp_hidden_layers.append(out_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.in_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            coord_dim=coord_dim,
            attention_type=gno_config.attention_type
        )

        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=1
        )

        if gno_config.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=gno_config.gno_coord_dim,
                output_dim=out_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                n_layers=1
            )
        
        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.coord_dim = coord_dim
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(self, pndata: torch.Tensor, x_coord: torch.Tensor, token_coord: torch.Tensor, encoder_nbrs: list):
        """
        pndata: [n_batch, n, n_channels]
        x_coord: [n_batch, n, d]
        token_coord: [n_batch, m, d]
        encoder_nbrs: list of length n_batch, each element is a list of neighbors for different scales
        """
        x = rescale(x_coord)
        device = pndata.device
        latent_queries = token_coord.to(device)
        pndata = pndata.permute(0,2,1)
        pndata = self.lifting(pndata).permute(0,2,1)
        
        n_batch, n, d = x_coord.shape
        # m = latent_queries.shape[1]
        #radii = minimal_support(latent_queries)
        encoded = []

        if self.use_scale_weights:
            scale_weights = self.scale_weighting(latent_queries) # [m, num_scales]
            scale_weights = self.scale_weight_activation(scale_weights) # [m, num_scales]
        for b in range(n_batch):
            x_b = x[b] # Shape: [n, d]
            pndata_b = pndata[b] # Shape: [n, n_channels]
    
            encoded_scales = []
            for scale_idx, scale_nbrs in enumerate(encoder_nbrs[b]):
                #scaled_radius = radii * scale
                # scaled_radius = self.gno_radius * scale
                # with torch.no_grad():
                #     spatial_nbrs = self.nb_search(x_b, latent_queries, scaled_radius)
                encoded_unpatched = self.gno(
                    y = x_b,
                    x = latent_queries,
                    f_y = pndata_b,
                    neighbors = scale_nbrs
                )

                if hasattr(self, 'geoembed'):
                    geoembedding = self.geoembed(
                        x_b,
                        latent_queries,
                        scale_nbrs
                    ) # Shape: [n, d]
                    encoded_unpatched = torch.cat([encoded_unpatched, geoembedding], dim=-1).unsqueeze(0)
                    encoded_unpatched = encoded_unpatched.permute(0, 2, 1)
                    encoded_unpatched = self.recovery(encoded_unpatched).permute(0, 2, 1).squeeze(0)
                encoded_scales.append(encoded_unpatched)
            
            if len(encoded_scales) == 1:
                encoded_data = encoded_scales[0]
            else:
                if self.use_scale_weights:
                    encoded_scales_stack = torch.stack(encoded_scales, dim=0) # # [num_scales, m, n_channels]
                    weights = scale_weights.unsqueeze(-1)
                    encoded_data = (encoded_scales_stack * weights.permute(1, 0, 2)).sum(dim=0)  # [m, n_channels]
                else:
                    encoded_data = torch.stack(encoded_scales, 0).sum(dim=0)
            
            encoded.append(encoded_data)
        encoded = torch.stack(encoded, 0) # Shape: [n_batch, m, n_channels]
        return encoded


############
# GNO Decoder
############
class GNODecoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)
        self.gno_radius = gno_config.gno_radius
        self.graph_cache = None
        self.use_graph_cache = gno_config.use_graph_cache
        self.scales = gno_config.scales
        self.use_scale_weights = gno_config.use_scale_weights

        out_kernel_in_dim = gno_config.gno_coord_dim * 2
        coord_dim = gno_config.gno_coord_dim 
        if gno_config.node_embedding:
            out_kernel_in_dim = gno_config.gno_coord_dim * 4 * 2 * 2  # 32
            coord_dim = gno_config.gno_coord_dim * 4 * 2
        if gno_config.out_gno_transform_type == "nonlinear" or gno_config.out_gno_transform_type == "nonlinear_kernelonly":
            out_kernel_in_dim += out_channels

        out_gno_channel_mlp_hidden_layers = gno_config.out_gno_channel_mlp_hidden_layers.copy()
        out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_channel_mlp_hidden_layers.append(in_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.out_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            coord_dim=coord_dim,
            attention_type=gno_config.attention_type
        )

        self.projection = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=gno_config.projection_channels,
            n_layers=2,
            n_dim=1,
        )

        if gno_config.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=gno_config.gno_coord_dim,
                output_dim=in_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                n_layers=1
            )

        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.coord_dim = coord_dim
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)  

    def forward(self, rndata: torch.Tensor, x_coord: torch.Tensor, token_coord: torch.Tensor, decoder_nbrs: list):
        """
        rndata: [n_batch, n, n_channels]
        x_coord: [n_batch, m, d]
        token_coord: [n_batch, n, d]
        decoder_nbrs: list of length n_batch, each element is a list of neighbors for different scales
        """
        device = rndata.device

        x = token_coord.to(device)
        latent_queries = rescale(x_coord)
        

        n_batch, n, d = latent_queries.shape
        
        decoded = []

        if self.use_scale_weights:
            scale_weights = self.scale_weighting(latent_queries) # [m, num_scales]
            scale_weights = self.scale_weight_activation(scale_weights) # [m, num_scales]
        for b in range(n_batch):
            latent_queries_b = latent_queries[b] # Shape: [m, d]

            rndata_b = rndata[b] # Shape: [n, n_channels]
            decoded_scales = []
            #radii = minimal_support(latent_queries_b)
            for scale_idx, scale_nbrs in enumerate(decoder_nbrs[b]):
                # scaled_radius = self.gno_radius * scale
                # #scaled_radius = radii * scale
                # with torch.no_grad():
                #     spatial_nbrs = self.nb_search(x, latent_queries_b, scaled_radius)
                decoded_unpatched = self.gno(
                    y = x,
                    x = latent_queries_b,
                    f_y = rndata_b,
                    neighbors = scale_nbrs
                )

                if hasattr(self, 'geoembed'):
                    geoembedding = self.geoembed(
                        x,
                        latent_queries_b,
                        scale_nbrs
                    )
                    decoded_unpatched = torch.cat([decoded_unpatched, geoembedding], dim=-1).unsqueeze(0)
                    decoded_unpatched = decoded_unpatched.permute(0, 2, 1)
                    decoded_unpatched = self.recovery(decoded_unpatched).permute(0, 2, 1).squeeze(0)
                decoded_scales.append(decoded_unpatched)
            
            if len(decoded_scales) == 1:
                decoded_data = decoded_scales[0]
            else:
                if self.use_scale_weights:
                    decoded_scales_stack = torch.stack(decoded_scales, dim=0)
                    weights = scale_weights[b].unsqueeze(-1)
                    decoded_data = (decoded_scales_stack * weights.permute(1, 0, 2)).sum(dim=0)
                else:
                    decoded_data = torch.stack(decoded_scales, 0).sum(dim=0)
            
            decoded.append(decoded_data)
        decoded = torch.stack(decoded, 0) # Shape: [n_batch, m, n_channels]
        decoded = decoded.permute(0,2,1)
        decoded = self.projection(decoded).permute(0, 2, 1)
        return decoded
