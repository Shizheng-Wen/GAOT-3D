import os 
import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader 
from torch_geometric.transforms import Compose
import torch_geometric as pyg

import numpy as np

from .base import TrainerBase
from .utils.metric import compute_drivaernet_metric
from .utils.plot import plot_3d_comparison_pyvista, plot_3d_comparison_matplotlib

from src.data.pyg_datasets import VTKMeshDataset, EnrichedData
from src.data.pyg_transforms import RescalePosition, RescalePositionNew, NormalizeFeatures
from src.model import init_model
from tqdm import tqdm

from src.utils.scale import rescale_new, rescale

EPSILON = 1e-10

def move_to_device(data, device):
    """Recursively move all tensors in a nested structure to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data

class StaticTrainer3D(TrainerBase):

    """
    Trainer for static problems, i.e. problems that do not depend on time.
    """

    def __init__(self, args):
        super().__init__(args)

    def _calculate_or_load_stats(self, dataset_config, order_file_path, data_root):
        """Calculates or loads normalization statistics."""
        stats_file = os.path.join(data_root, f"{dataset_config.name}_norm_stats.pt")

        if os.path.exists(stats_file) and not getattr(dataset_config, 'force_recompute_stats', False):
            print(f"Loading pre-calculated normalization stats from {stats_file}")
            stats = torch.load(stats_file)
            self.u_mean = stats['mean'].to(self.dtype)
            self.u_std = stats['std'].to(self.dtype)
        else:
            print("Calculating normalization statistics from training set...")
            # Need to instantiate a temporary dataset for the training split *without* normalization
            # to iterate and compute stats.
            temp_train_dataset = VTKMeshDataset(
                root=dataset_config.base_path,
                order_file=order_file_path,
                dataset_config=dataset_config,
                split='train',
                transform=RescalePosition() # Apply rescaling even when calculating stats
            )
            # Use a simple loader to iterate
            temp_loader = PyGDataLoader(temp_train_dataset, batch_size=dataset_config.batch_size, shuffle=False, num_workers=4) # Use 0 workers for simplicity here

            all_x = []
            for batch in tqdm(temp_loader, desc="Calculating Stats"):
                # Collect all 'x' features from the training set
                # Be mindful of memory for large datasets!
                all_x.append(batch.x.cpu()) # Move to CPU to avoid GPU memory buildup

            if not all_x:
                 raise ValueError("No data found in training set to calculate statistics.")

            full_x_tensor = torch.cat(all_x, dim=0)
            # Calculate mean/std over all nodes/samples in training set (dimension 0)
            # Assuming features are [TotalNodes, NumFeatures]
            
            self.u_mean = torch.mean(full_x_tensor, dim=0, dtype=self.dtype)
            self.u_std = torch.std(full_x_tensor, dim=0).to(self.dtype)

            print(f"Calculated - Mean: {self.u_mean}, Std: {self.u_std}")
            # Save calculated stats
            print(f"Saving normalization stats to {stats_file}")
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            torch.save({'mean': self.u_mean, 'std': self.u_std}, stats_file)
        # Ensure stats are available and have correct type
        if self.u_mean is None or self.u_std is None:
             raise RuntimeError("Normalization mean/std could not be calculated or loaded.")
        self.u_mean = self.u_mean.to(self.dtype)
        self.u_std = self.u_std.to(self.dtype)
   
    def _update_pt_files_with_edges(self, dataset_config, order_file_path, gno_config):
        """
        Iterates through processed .pt files, computes edges (and optionally counts),
        and saves them back to the files. Runs ONLY on rank 0 in DDP.
        """
        if self.setup_config.rank != 0:
            return
        
        processed_dir = os.path.join(dataset_config.base_path, dataset_config.processed_folder)
        if not os.path.isdir(processed_dir):
             raise FileNotFoundError(f"Processed directory for update not found: {processed_dir}")
        
        with open(order_file_path, 'r') as f:
            all_filenames_base = [line.strip() for line in f if line.strip()]
        total_samples = len(all_filenames_base)
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        # Test size calculation depends on how split was done previously
        test_size = dataset_config.test_size
        indices = np.arange(total_samples) 
        # Note: rand_dataset shuffling is NOT applied here, we process based on original order file names
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[-test_size:] 
        relevant_indices = np.concatenate([train_indices, val_indices, test_indices])
        filenames_to_process = [f"{all_filenames_base[i]}.pt" for i in relevant_indices]

        
        print(f"Rank 0: Checking/Updating {len(filenames_to_process)} '.pt' files in {processed_dir} with edge information...")

        latent_tokens_cpu = self.latent_tokens.cpu()
        num_latent_tokens = latent_tokens_cpu.shape[0]

        from src.model.layers.magno import get_neighbor_strategy
        for filename in tqdm(filenames_to_process, desc="Updating .pt files"):
            fpath = os.path.join(processed_dir, filename)
            fpath_tmp = fpath + ".tmp"
            try:
                data_old = torch.load(fpath, map_location='cpu')
                data = EnrichedData(pos = data_old.pos, x=data_old.x)
                for attr, value in data_old:
                     if attr not in ['pos', 'x']: 
                         setattr(data, attr, value)
                data.num_latent_nodes = num_latent_tokens
                if dataset_config.use_rescale_new:
                    phys_pos = rescale_new(data.pos.to(torch.float), (-1, 1), self.metadata.domain_x)
                else:
                    phys_pos = rescale(data.pos.to(torch.float), (-1, 1)) # Rescale to [-1, 1], very important!
                num_physical_nodes = phys_pos.shape[0]
                batch_idx_phys = torch.zeros(num_physical_nodes, dtype=torch.long)
                batch_idx_latent = torch.zeros(num_latent_tokens, dtype=torch.long)

                for scale_idx, scale in enumerate(gno_config.scales):
                    scaled_radius = gno_config.gno_radius * scale
                    # --- Encoder Edges ---
                    enc_edge_index = get_neighbor_strategy(
                        neighbor_strategy=gno_config.neighbor_strategy,
                        phys_pos = phys_pos,
                        batch_idx_phys = batch_idx_phys,
                        latent_tokens_pos = latent_tokens_cpu,
                        batch_idx_latent = batch_idx_latent,
                        radius=scaled_radius
                    ).to(torch.int32)
                    setattr(data, f'encoder_edge_index_s{scale_idx}', enc_edge_index)
                    if enc_edge_index.numel() > 0:
                        enc_counts = torch.bincount(enc_edge_index[0], minlength=num_latent_tokens).to(torch.int32)
                    else:
                        enc_counts = torch.zeros(num_latent_tokens, dtype=torch.int32)
                    setattr(data, f'encoder_query_counts_s{scale_idx}', enc_counts)
                    # --- Decoder Edges ---
                    dec_edge_index = get_neighbor_strategy(
                        neighbor_strategy=gno_config.neighbor_strategy,
                        phys_pos = latent_tokens_cpu,
                        batch_idx_phys = batch_idx_latent,
                        latent_tokens_pos = phys_pos,
                        batch_idx_latent = batch_idx_phys,
                        radius=scaled_radius
                    ).to(torch.int32)
                    setattr(data, f'decoder_edge_index_s{scale_idx}', dec_edge_index)
                    if dec_edge_index.numel() > 0:
                        dec_counts = torch.bincount(dec_edge_index[0], minlength=num_physical_nodes).to(torch.int32)
                    else:
                        dec_counts = torch.zeros(num_physical_nodes, dtype=torch.int32)
                    setattr(data, f'decoder_query_counts_s{scale_idx}', dec_counts)
                # --- Save updated data ---
                torch.save(data, fpath_tmp)
                os.replace(fpath_tmp, fpath)
            except FileNotFoundError:
                print(f"Warning: File not found during update: {fpath}")
            except Exception as e:
                print(f"Error processing file {fpath}: {e}")
                if os.path.exists(fpath_tmp):
                     os.remove(fpath_tmp)
        print(f"Rank 0: Finished checking/updating '.pt' files.")
            
    def init_dataset(self, dataset_config):
        print("Initializing PyG dataset ...")  
        self.latent_token_size = self.model_config.args.latent_tokens
        data_root = dataset_config.base_path
        order_file_path = os.path.join(data_root, "order_use.txt")
        processed_data_path = os.path.join(data_root, dataset_config.processed_folder)

        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_path}")
        if not os.path.exists(order_file_path):
            raise FileNotFoundError(f"Order file does not exist: {order_file_path}")

        # --- Latent Tokens ---
        phy_domain = self.metadata.domain_x
        x_min, y_min, z_min = phy_domain[0]
        x_max, y_max, z_max = phy_domain[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, self.latent_token_size[0]),
            torch.linspace(y_min, y_max, self.latent_token_size[1]),
            torch.linspace(z_min, z_max, self.latent_token_size[2]),
            indexing = "ij"
        )
        latent_queries = torch.stack(meshgrid, dim = -1).reshape(-1, 3)
        if dataset_config.use_rescale_new:
            self.latent_tokens = rescale_new(latent_queries, (-1, 1), phy_domain)
        else:
            self.latent_tokens = rescale(latent_queries, (-1, 1))
        
        print(f"Rank {self.setup_config.rank}: Dataset initialization finished.")

        # --- Pre-computation /Update Step ---
        if dataset_config.update_pt_files_with_edges:
            gno_cfg_for_precompute = self.model_config.args.magno
            if self.setup_config.rank == 0:
                self._update_pt_files_with_edges(
                    dataset_config,
                    order_file_path,
                    gno_cfg_for_precompute
                )
            if self.setup_config.distributed:
                print(f"Rank {self.setup_config.rank}: Waiting for edge pre-computation barrier...")
                torch.distributed.barrier()
                print(f"Rank {self.setup_config.rank}: Barrier passed.")
            
            self.model_config.args.magno.precompute_edges = True
            print(f"Rank {self.setup_config.rank}: Set model config to use precomputed edges.")

        # --- end Pre-computation /Update Step ---
            
        # --- Calculate or Load Normalization Stats ---
        self._calculate_or_load_stats(dataset_config, order_file_path, data_root)
        

        # --- Define Transforms ---
        if dataset_config.use_rescale_new:
            rescale_transform = RescalePositionNew(lims=(-1., 1.), phy_domain = phy_domain)
        else:
            rescale_transform = RescalePosition(lims=(-1., 1.))
        
        normalize_transform = NormalizeFeatures(mean=self.u_mean, std=self.u_std)
        composed_transform = Compose([rescale_transform, normalize_transform])

        if self.setup_config.train:
            print(f"Rank {self.setup_config.rank}: Loading train dataset...")
            train_ds = VTKMeshDataset(
                root = data_root,
                order_file = order_file_path,
                dataset_config = dataset_config,
                split="train",
                transform = composed_transform
            )
            print(f"Rank {self.setup_config.rank}: Loading validation dataset...")
            val_ds = VTKMeshDataset(
                root = data_root,
                order_file = order_file_path,
                dataset_config = dataset_config,
                split="val",
                transform = composed_transform
            )
        print(f"Rank {self.setup_config.rank}: Loading test dataset...")
        test_ds = VTKMeshDataset(
            root = data_root,
            order_file = order_file_path,
            dataset_config = dataset_config,
            split="test",
            transform = composed_transform
        )

        self.num_input_channels = test_ds[0].pos.shape[1]
        self.num_output_channels = test_ds[0].x.shape[1]
        
        # --- Create DataLoaders ---
        if self.setup_config.train:
            ## --- Create DistributedSampler for training ---
            train_sampler = None
            if self.setup_config.distributed:
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=self.setup_config.world_size,
                    rank=self.setup_config.rank,
                    shuffle=dataset_config.shuffle, # Sampler handles shuffling
                    drop_last=True # Often good practice for DDP
                )
                print(f"Rank {self.setup_config.rank}: Created DistributedSampler for training.")

            self.train_loader = PyGDataLoader(
                train_ds,
                batch_size=dataset_config.batch_size,
                shuffle=False if train_sampler is not None else dataset_config.shuffle,
                num_workers=dataset_config.num_workers,
                sampler=train_sampler,
                pin_memory=True, # Good practice if using GPU
                drop_last=train_sampler is not None 
            )

            val_sampler = None
            # if self.setup_config.distributed:
            #     val_sampler = DistributedSampler(val_ds, shuffle=False, rank=self.setup_config.rank)
            self.val_loader = PyGDataLoader(
                val_ds,
                batch_size=dataset_config.batch_size,
                shuffle=False,
                num_workers=dataset_config.num_workers,
                pin_memory=True,
                sampler=val_sampler
            )

        test_sampler = None
        # if self.setup_config.distributed:
        #     test_sampler = DistributedSampler(test_ds, shuffle=False, rank=self.setup_config.rank)
        self.test_loader = PyGDataLoader(   
            test_ds,
            batch_size=dataset_config.batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
            sampler=test_sampler
        )

    def init_model(self, model_config):
        self.model = init_model(
            input_size=self.num_input_channels, 
            output_size=self.num_output_channels, 
            model=model_config.name,
            config=model_config.args
            )

        self.model.to(self.device)
        
        if self.setup_config.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.setup_config.local_rank],
                output_device=self.setup_config.local_rank
            )
        
    def train_step(self, batch: "pyg.data.Batch") -> torch.Tensor:
        batch = batch.to(self.device)
        latent_tokens_dev = self.latent_tokens.to(self.device)

        pred = self.model(
            batch = batch,
            tokens_pos = latent_tokens_dev
        )

        target = batch.x
        target = target
        return self.loss_fn(pred, target)

    def validate(self, loader: PyGDataLoader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                latent_tokens_dev = self.latent_tokens.to(self.device)
                pred = self.model(
                    batch = batch,
                    tokens_pos = latent_tokens_dev
                )
                target = batch.x
                target = target

                loss = self.loss_fn(pred, target)
                total_loss += loss.item()
        return total_loss / len(loader.dataset)

    def test(self):
        self.model.eval()
        metric_suite = self.dataset_config.metric_suite
        
        # Store data needed for metric calculation and plotting
        all_batch_targets_denorm = []
        all_batch_preds_denorm = []
        plot_coords, plot_gtr, plot_prd = None, None, None # For plotting first sample
        
        print(f"Starting testing with metric suite: '{metric_suite}'")

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                batch = batch.to(self.device)
                latent_tokens_dev = self.latent_tokens.to(self.device)

                pred_norm = self.model(batch, latent_tokens_dev)
                target_norm = batch.x

                # De-normalize predictions and targets
                u_std_dev = self.u_std.to(self.device)
                u_mean_dev = self.u_mean.to(self.device)
                pred_de_norm = pred_norm * u_std_dev + u_mean_dev
                target_de_norm = target_norm * u_std_dev + u_mean_dev

                # --- Store results for aggregation ---
                # Store de-normalized results on CPU to save GPU memory
                all_batch_targets_denorm.append(target_de_norm.cpu())
                all_batch_preds_denorm.append(pred_de_norm.cpu())
                # --- End Storing ---

                # --- Plotting Logic (extract data for first sample of first batch) ---
                if i == 0 and self.setup_config.rank == 0:
                    plotting_idx = 0
                    first_graph_node_mask = (batch.batch == plotting_idx)
                    plot_coords = batch.pos[first_graph_node_mask].cpu().numpy()
                    plot_gtr = target_de_norm[first_graph_node_mask].cpu().numpy()
                    plot_prd = pred_de_norm[first_graph_node_mask].cpu().numpy()
                    print(f"Extracted plotting data of {batch.filename[plotting_idx]}:"
                            f"coords shape {plot_coords.shape}, gtr shape {plot_gtr.shape}, prd shape {plot_prd.shape}"
                        )
                # --- End Plotting Logic ---

        # --- Aggregate Metrics and Plot (Rank 0 only) ---
        if self.setup_config.rank == 0:
            full_preds = torch.cat(all_batch_preds_denorm, dim=0)
            full_targets = torch.cat(all_batch_targets_denorm, dim=0)
            print(f"Concatenated results: preds shape {full_preds.shape}, targets shape {full_targets.shape}")

            # --- Calculate Metrics ---
            if metric_suite == "poseidon":
                print("Warning: 'poseidon' metric suite requires adaptation for variable nodes per sample PyG structure. Skipping calculation.")
                final_metric = float('nan')
                self.config.datarow["relative error (direct)"] = final_metric # Store NaN
            
            if metric_suite == "drivaernet":
                agg_metrics = compute_drivaernet_metric(
                    gtr_ls = all_batch_targets_denorm,
                    prd_ls = all_batch_preds_denorm,
                    metadata =  self.metadata
                )

                print(f"--- Final Metrics (Drivaernet Suite - Full Dataset) ---")
                print(f"MSE (x10^-2):       {agg_metrics['MSE'] * 100:.4f}")
                print(f"MAE (x10^-1):       {agg_metrics['MAE'] * 10:.4f}")
                print(f"RMSE:               {agg_metrics['RMSE']:.4f}")
                print(f"Max_Error:          {agg_metrics['Max_Error']:.4f}")
                print(f"Rel L2 Error (%):   {agg_metrics['Rel_L2']:.4f}")
                print(f"Rel L1 Error (%):   {agg_metrics['Rel_L1']:.4f}")

            elif metric_suite == "general":
                # Calculate general metrics on the full concatenated tensors
                diff = full_preds - full_targets
                mse = torch.mean(diff ** 2).item()
                mae = torch.mean(torch.abs(diff)).item()
                max_ae = torch.max(torch.abs(diff)).item()

                norm_diff_l2 = torch.linalg.norm(diff.float()) 
                norm_gtr_l2 = torch.linalg.norm(full_targets.float())
                rel_l2 = (norm_diff_l2 / (norm_gtr_l2 + EPSILON)).item() * 100.0

                norm_diff_l1 = torch.linalg.norm(diff.float(), ord=1)
                norm_gtr_l1 = torch.linalg.norm(full_targets.float(), ord=1)
                rel_l1 = (norm_diff_l1 / (norm_gtr_l1 + EPSILON)).item() * 100.0

                final_metrics_dict = {
                    'MSE': mse, 'MAE': mae, 'Max AE': max_ae,
                    'Rel L2 Error (%)': rel_l2, 'Rel L1 Error (%)': rel_l1
                }

                # Store and print results
                self.config.datarow["MSE (x10^-2)"] = final_metrics_dict['MSE'] * 100
                self.config.datarow["MAE (x10^-1)"] = final_metrics_dict['MAE'] * 10
                self.config.datarow["Max AE"] = final_metrics_dict['Max AE']
                self.config.datarow["Rel L2 Error (%)"] = final_metrics_dict['Rel L2 Error (%)']
                self.config.datarow["Rel L1 Error (%)"] = final_metrics_dict['Rel L1 Error (%)']

                print(f"--- Final Metrics (General Suite - Full Dataset) ---")
                print(f"MSE (x10^-2):       {final_metrics_dict['MSE'] * 100:.4f}")
                print(f"MAE (x10^-1):       {final_metrics_dict['MAE'] * 10:.4f}")
                print(f"Max AE:             {final_metrics_dict['Max AE']:.4f}")
                print(f"Rel L2 Error (%):   {final_metrics_dict['Rel L2 Error (%)']:.4f}")
                print(f"Rel L1 Error (%):   {final_metrics_dict['Rel L1 Error (%)']:.4f}")

            # --- Plotting the stored first sample ---
            print("Attempting to plot first sample...")
            try:
                channel_to_plot = 0
                gtr_np = plot_gtr[:, channel_to_plot]
                prd_np = plot_prd[:, channel_to_plot]

                plot_save_path = self.path_config.result_path
                var_name = self.metadata.names['u'] 
                plot_3d_comparison_matplotlib(
                    coords=plot_coords, u_gtr=gtr_np, u_prd=prd_np,
                    save_path=plot_save_path,
                    variable_name=var_name,
                    point_size=2.0, view_angle=(25, -135)
                )
                print(f"Saved plot for first test sample to {plot_save_path}")
            except Exception as plot_err:
                print(f"Error during 3D plotting of first test sample: {plot_err}")

        elif self.setup_config.distributed:
             pass 