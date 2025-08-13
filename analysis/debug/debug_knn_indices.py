#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your project modules
from src.data.pyg_datasets import VTKMeshDataset
from src.model.layers.magno import get_neighbor_strategy
from torch_geometric.nn import knn
from src.trainer.utils.default_set import DatasetConfig, merge_config

def debug_knn_indices():
    """Debug k-NN index generation for problematic samples"""
    
    # Load problematic samples
    problematic_samples = ['boundary_102']

    config_path = "config/drivaerml/pressure.json"
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    dataset_config = merge_config(DatasetConfig, config['dataset'])
    
    for sample_name in problematic_samples:
        print(f"\n=== Debugging {sample_name} ===")
        
        try:
            # Create dataset 
            dataset = VTKMeshDataset(
                root=dataset_config.base_path,
                order_file=os.path.join(dataset_config.base_path, 'order_use.txt'),
                dataset_config=dataset_config,
                split='train'
            )
            
            print(f"Dataset size: {len(dataset)}")

            # Get the sample - we'll use the first sample which should be boundary_102
            sample = dataset[0]
            print(f"Sample filename: {sample.filename}")
            print(f"Physical nodes: {sample.x.shape[0]}")
            print(f"Physical positions shape: {sample.pos.shape}")
            print(f"Num latent nodes: {sample.num_latent_nodes}")
            
            # Extract positions - ignore precomputed edges
            phys_pos = sample.pos
            num_phys_nodes = phys_pos.shape[0]
            num_latent_nodes = sample.num_latent_nodes
            
            print(f"Physical pos shape: {phys_pos.shape}")
            print(f"Physical pos device: {phys_pos.device}")
            print(f"Physical pos dtype: {phys_pos.dtype}")
            
            # Create batch indices for physical nodes (assuming single batch)
            batch_idx_phys = torch.zeros(num_phys_nodes, dtype=torch.long, device=phys_pos.device)
            
            # Create latent token positions (simulating what would be in the actual model)
            # We'll create them within the same spatial bounds as physical positions
            pos_min = phys_pos.min(dim=0)[0]
            pos_max = phys_pos.max(dim=0)[0]
            
            print(f"Physical position bounds:")
            print(f"  Min: {pos_min}")
            print(f"  Max: {pos_max}")
            
            # Create latent positions uniformly distributed in the same space
            latent_tokens_pos = torch.rand(num_latent_nodes, phys_pos.shape[1], device=phys_pos.device, dtype=phys_pos.dtype)
            latent_tokens_pos = pos_min + latent_tokens_pos * (pos_max - pos_min)
            
            # Create batch indices for latent tokens (assuming single batch)
            latent_tokens_batch_idx = torch.zeros(num_latent_nodes, dtype=torch.long, device=phys_pos.device)
            
            print(f"Latent pos shape: {latent_tokens_pos.shape}")
            print(f"Latent pos device: {latent_tokens_pos.device}")
            print(f"Latent pos dtype: {latent_tokens_pos.dtype}")
            
            print(f"Physical batch shape: {batch_idx_phys.shape}")
            print(f"Latent batch shape: {latent_tokens_batch_idx.shape}")
            
            # Test k-NN with different k values
            for k in [1, 2, 4]:
                print(f"\n  Testing k-NN with k={k}")
                try:
                    # Test the k-NN function directly
                    edge_index = knn(
                        x=latent_tokens_pos,      # Data points to search within (latent)
                        y=phys_pos,               # Query points (physical)
                        k=k,                      # Number of nearest neighbors
                        batch_x=latent_tokens_batch_idx, # Batch index for data points
                        batch_y=batch_idx_phys    # Batch index for query points
                    )
                    
                    print(f"    k-NN successful with k={k}")
                    print(f"    Edge index shape: {edge_index.shape}")
                    print(f"    Expected edges: {phys_pos.shape[0] * k}")
                    print(f"    Actual edges: {edge_index.shape[1]}")
                    
                    # Check index ranges
                    phys_indices = edge_index[0]
                    latent_indices = edge_index[1]
                    
                    print(f"    Physical indices range: {phys_indices.min().item()} to {phys_indices.max().item()}")
                    print(f"    Latent indices range: {latent_indices.min().item()} to {latent_indices.max().item()}")
                    
                    # Check if indices are within bounds
                    max_phys_idx = phys_pos.shape[0] - 1
                    max_latent_idx = latent_tokens_pos.shape[0] - 1
                    
                    if phys_indices.max().item() > max_phys_idx:
                        print(f"    ERROR: Physical index out of bounds! Max: {phys_indices.max().item()}, Should be <= {max_phys_idx}")
                    
                    if latent_indices.max().item() > max_latent_idx:
                        print(f"    ERROR: Latent index out of bounds! Max: {latent_indices.max().item()}, Should be <= {max_latent_idx}")
                        
                    if phys_indices.min().item() < 0:
                        print(f"    ERROR: Physical index negative! Min: {phys_indices.min().item()}")
                        
                    if latent_indices.min().item() < 0:
                        print(f"    ERROR: Latent index negative! Min: {latent_indices.min().item()}")
                    
                    # Test indexing operations that would happen in forward pass
                    try:
                        selected_phys_pos = phys_pos[phys_indices]
                        selected_latent_pos = latent_tokens_pos[latent_indices]
                        print(f"    Indexing test passed for k={k}")
                        print(f"    Selected physical pos shape: {selected_phys_pos.shape}")
                        print(f"    Selected latent pos shape: {selected_latent_pos.shape}")
                    except Exception as e:
                        print(f"    ERROR: Indexing test failed for k={k}: {e}")
                    
                except Exception as e:
                    print(f"    ERROR: k-NN failed with k={k}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Test our modified get_neighbor_strategy function with decoder mode
            print(f"\n  Testing get_neighbor_strategy function (decoder mode)")
            try:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy='radius',  # Parameter doesn't matter in decoder mode
                    phys_pos=phys_pos,
                    batch_idx_phys=batch_idx_phys,
                    latent_tokens_pos=latent_tokens_pos,
                    batch_idx_latent=latent_tokens_batch_idx,
                    radius=0.033,
                    k_neighbors=2,
                    is_decoder=True
                )
                
                print(f"    get_neighbor_strategy successful")
                print(f"    Edge index shape: {edge_index.shape}")
                
                # Check if indices are within bounds
                if edge_index.shape[1] > 0:
                    source_indices = edge_index[0]  # Should be latent indices  
                    target_indices = edge_index[1]  # Should be physical indices
                    
                    print(f"    Source (latent) indices range: {source_indices.min().item()} to {source_indices.max().item()}")
                    print(f"    Target (physical) indices range: {target_indices.min().item()} to {target_indices.max().item()}")
                    
                    # Test the indexing that would happen in the forward pass
                    try:
                        y_pos_indexed = latent_tokens_pos[source_indices]  # This is y_pos[source_idx] in the error
                        x_pos_indexed = phys_pos[target_indices]
                        print(f"    Final indexing test passed")
                        print(f"    y_pos_indexed shape: {y_pos_indexed.shape}")
                        print(f"    x_pos_indexed shape: {x_pos_indexed.shape}")
                    except Exception as e:
                        print(f"    ERROR: Final indexing test failed: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"    ERROR: get_neighbor_strategy failed: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"ERROR loading sample {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    debug_knn_indices() 