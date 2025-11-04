"""
Unified collate functions with online graph building.
Handles batch processing and PyG-style format conversion with on-demand graph construction.
"""
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from torch_geometric.data import Batch

from src.model.layers.magno import get_neighbor_strategy, parse_neighbor_strategy
from src.data.pyg_datasets import EnrichedData, VTKMeshDataset


logger = logging.getLogger(__name__)

class UnifiedCollateFunction:
    """
    Unified collate function that handles online graph building.
    """
    
    def __init__(self, 
                 coord_dim: int = 2,
                 magno_radius: float = 0.033,
                 magno_scales: List[float] = [1.0],
                 latent_tokens: torch.Tensor = None,
                 neighbor_search_method: str = "bidirectional",
                 k_neighbors: int = 1,
                 asynchronous_graph_building: bool = True):
        """
        Initialize unified collate function.
        
        Args:
            coord_dim: Coordinate dimension (2 or 3)
            magno_radius: Base radius for graph neighbor search
            magno_scales: List of scale factors for multi-scale graphs
            latent_tokens: Latent tokens coordinates
            neighbor_search_method: Method for neighbor search (bidirectional, radius, knn)
            k_neighbors: Number of neighbors for neighbor search
            asynchronous_graph_building: Whether to build graphs online
        """
        self.coord_dim = coord_dim
        self.magno_radius = magno_radius
        self.magno_scales = magno_scales
        self.neighbor_search_method = neighbor_search_method
        self.k_neighbors = k_neighbors
        self.asynchronous_graph_building = asynchronous_graph_building
        
        self.latent_queries = latent_tokens

    def __call__(self, batch: List[VTKMeshDataset]) -> Batch:
        f"""
        Collate function that processes a batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batch of samples
        """        
        data_list: List[EnrichedData] = []
        for data in batch:
            if not isinstance(data, EnrichedData):
                enriched = EnrichedData(pos=data.pos, x=data.x)
                # Copy over all existing attributes except ones we explicitly set
                for attr, value in data:
                    if attr not in ['pos', 'x']:
                        setattr(enriched, attr, value)
                data = enriched
            data_list.append(data)

        # Always rebuild if online_graph_building is True
        if self.asynchronous_graph_building:
            # Prepare strategies
            enc_strategy, dec_strategy = parse_neighbor_strategy(self.neighbor_search_method)
            latent_tokens = self.latent_queries
            if latent_tokens is None:
                raise ValueError("latent_tokens must be provided for online graph building in collate function.")
            num_latent_nodes = int(latent_tokens.shape[0])

            # Ensure CPU tensors for neighbor search in workers
            latent_tokens_cpu = latent_tokens.cpu()

            for data in data_list:
                # Ensure attribute for batching increments
                data.num_latent_nodes = num_latent_nodes

                phys_pos = data.pos.to(torch.float32)
                num_phys_nodes = int(phys_pos.shape[0])
                batch_idx_phys = torch.zeros(num_phys_nodes, dtype=torch.long)
                batch_idx_latent = torch.zeros(num_latent_nodes, dtype=torch.long)

                for scale_idx, scale in enumerate(self.magno_scales):
                    scaled_radius = float(self.magno_radius) * float(scale)

                    # Encoder edges: phys -> latent, edge_index [2, E] = [phys_idx, latent_idx]
                    enc_edge_index = get_neighbor_strategy(
                        neighbor_strategy=enc_strategy,
                        phys_pos=phys_pos,
                        batch_idx_phys=batch_idx_phys,
                        latent_tokens_pos=latent_tokens_cpu,
                        batch_idx_latent=batch_idx_latent,
                        radius=scaled_radius,
                        k_neighbors=int(self.k_neighbors),
                        is_decoder=False
                    ).to(dtype=torch.long)
                    setattr(data, f'encoder_edge_index_s{scale_idx}', enc_edge_index)
                    if enc_edge_index.numel() > 0:
                        enc_counts = torch.bincount(enc_edge_index[1], minlength=num_latent_nodes).to(dtype=torch.long)
                    else:
                        enc_counts = torch.zeros(num_latent_nodes, dtype=torch.long)
                    setattr(data, f'encoder_query_counts_s{scale_idx}', enc_counts)

                    # Decoder edges: latent -> phys, edge_index [2, E] = [latent_idx, phys_idx]
                    dec_edge_index = get_neighbor_strategy(
                        neighbor_strategy=dec_strategy,
                        phys_pos=phys_pos,
                        batch_idx_phys=batch_idx_phys,
                        latent_tokens_pos=latent_tokens_cpu,
                        batch_idx_latent=batch_idx_latent,
                        radius=scaled_radius,
                        k_neighbors=int(self.k_neighbors),
                        is_decoder=True
                    ).to(dtype=torch.long)
                    setattr(data, f'decoder_edge_index_s{scale_idx}', dec_edge_index)
                    if dec_edge_index.numel() > 0:
                        dec_counts = torch.bincount(dec_edge_index[1], minlength=num_phys_nodes).to(dtype=torch.long)
                    else:
                        dec_counts = torch.zeros(num_phys_nodes, dtype=torch.long)
                    setattr(data, f'decoder_query_counts_s{scale_idx}', dec_counts)

        return Batch.from_data_list(data_list)

    def update_config(self, **kwargs):
        """
        Update collate function configuration.
        
        Useful for dynamic configuration changes during training.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown config parameter {key}")

def create_collate_function(coord_dim: int = 2, 
                           magno_radius: float = 0.033,
                           magno_scales: List[float] = [1.0],
                           latent_tokens: torch.Tensor = None,
                           neighbor_search_method: str = "bidirectional",
                           k_neighbors: int = 1,
                           asynchronous_graph_building: bool = True,
                           **kwargs) -> UnifiedCollateFunction:
    """
    Factory function to create a collate function with appropriate configuration.
    
    Args:
        coord_dim: Coordinate dimension
        magno_radius: Base graph radius
        magno_scales: Graph scale factors
        latent_tokens: Latent tokens coordinates
        neighbor_search_method: Method for neighbor search (bidirectional, radius, knn)
        k_neighbors: Number of neighbors for neighbor search
        asynchronous_graph_building: Whether to build graphs online
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured UnifiedCollateFunction instance
    """
    return UnifiedCollateFunction(
        coord_dim=coord_dim,
        magno_radius=magno_radius,
        magno_scales=magno_scales,
        latent_tokens=latent_tokens,
        neighbor_search_method=neighbor_search_method,
        k_neighbors=k_neighbors,
        asynchronous_graph_building=asynchronous_graph_building,
        **kwargs
    )
