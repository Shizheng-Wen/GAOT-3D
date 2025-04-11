import os 
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader 
from torch_geometric.data import Data 
from torch_geometric.transforms import Compose
import torch_geometric as pyg

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from .base import TrainerBase
from .utils.metric import compute_batch_errors, compute_final_metric 
from .utils.metric import compute_general_metrics_batch, aggregate_general_metrics
from .utils.plot import plot_3d_comparison_pyvista, plot_3d_comparison_matplotlib
from .utils.data_pairs import CustomDataset


from src.data.dataset import Metadata, DATASET_METADATA
from src.data.pyg_datasets import VTKMeshDataset
from src.data.pyg_transforms import RescalePosition, NormalizeFeatures
from src.model import init_model
from tqdm import tqdm
from src.model.layers.utils.magno_utils import NeighborSearch

from src.utils.scale import rescale
from src.utils.dataclass import shallow_asdict

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
   
    def init_dataset(self, dataset_config):
        print("Initializing PyG dataset ...")  
        self.latent_token_size = self.model_config.args.latent_tokens
        data_root = dataset_config.base_path
        order_file_path = os.path.join(data_root, "order_use.txt")
        processed_data_path = os.path.join(data_root, "processed_pyg")

        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_path}")
        if not os.path.exists(order_file_path):
            raise FileNotFoundError(f"Order file does not exist: {order_file_path}")
        
        # --- Calculate or Load Normalization Stats ---
        self._calculate_or_load_stats(dataset_config, order_file_path, data_root)

        # --- Define Transforms ---
        rescale_transform = RescalePosition(lims=(-1., 1.))
        normalize_transform = NormalizeFeatures(mean=self.u_mean, std=self.u_std)
        composed_transform = Compose([rescale_transform, normalize_transform])

        if self.setup_config.train:
            print("Loading training dataset ...")
            train_ds = VTKMeshDataset(
                root = data_root,
                order_file = order_file_path,
                dataset_config = dataset_config,
                split="train",
                transform = composed_transform
            )
            print("Loading validation dataset ...")
            val_ds = VTKMeshDataset(
                root = data_root,
                order_file = order_file_path,
                dataset_config = dataset_config,
                split="val",
                transform = composed_transform
            )
        print("Loading testing dataset ...")
        test_ds = VTKMeshDataset(
            root = data_root,
            order_file = order_file_path,
            dataset_config = dataset_config,
            split="test",
            transform = composed_transform
        )

        self.num_input_channels = 3
        self.num_output_channels = 1
        
        if self.setup_config.train:
            self.train_loader = PyGDataLoader(
                train_ds,
                batch_size=dataset_config.batch_size,
                shuffle=dataset_config.shuffle,
                num_workers=dataset_config.num_workers,
                pin_memory=True, # Good practice if using GPU
                # drop_last=True # Consider if needed for distributed training
            )
            self.val_loader = PyGDataLoader(
                val_ds,
                batch_size=dataset_config.batch_size,
                shuffle=False,
                num_workers=dataset_config.num_workers,
                pin_memory=True
            )

        self.test_loader = PyGDataLoader(
            test_ds,
            batch_size=dataset_config.batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True
        )

        # --- Latent Tokens (remains the same) ---
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
        self.latent_tokens = rescale(latent_queries, (-1, 1))
        print("Dataset initialization with PyG finished.")

    def init_model(self, model_config):
        self.model = init_model(
            input_size=self.num_input_channels, 
            output_size=self.num_output_channels, 
            model=model_config.name,
            config=model_config.args
            )
        breakpoint()
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
        self.model.to(self.device)
        metric_suite = self.dataset_config.metric_suite
        all_batch_metrics_data = []
        all_batch_coords = []
        all_batch_targets = []
        all_batch_preds = []
        
        print(f"Starting testing with metric suite: '{metric_suite}'")

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                batch = batch.to(self.device)
                latent_tokens_dev = self.latent_tokens.to(self.device)


                pred_norm = self.model(
                    batch,
                    latent_tokens_dev
                )
                target_norm = batch.x

                u_std_dev = self.u_std.to(self.device)
                u_mean_dev = self.u_mean.to(self.device)
                pred_de_norm = pred_norm * u_std_dev + u_mean_dev
                target_de_norm = target_norm * u_std_dev + u_mean_dev

                all_batch_targets.append(target_de_norm.cpu())
                all_batch_preds.append(pred_de_norm.cpu())

                if i == 0 and self.setup_config.rank == 0:
                    try:
                        first_graph_indices = torch.where(batch.batch == 0)[0]
                        plot_coords = batch.pos[first_graph_indices].cpu().numpy()
                        plot_gtr = target_de_norm[first_graph_indices].cpu().numpy()
                        plot_prd = pred_de_norm[first_graph_indices].cpu().numpy()
                    except Exception as e:
                        print(f"Error extracting first graph data for plotting: {e}")
                        plot_coords, plot_gtr, plot_prd = None, None, None

        if self.setup_config.rank == 0:
            if not all_batch_preds:
                print("Warning: No predictions collected during testing.")
                return

            if metric_suite == "poseidon":
                # Poseidon metric needs modification for PyG batch structure
                # It expects [B, T, S, V] and metadata. Adapting it requires care.
                # Placeholder: Indicate need for adaptation
                print("Warning: 'poseidon' metric suite requires adaptation for PyG data structure. Skipping.")
                final_metric = float('nan')
                self.config.datarow["relative error (direct)"] = final_metric

            elif metric_suite == "general":
                 # Compute general metrics on the concatenated full dataset results
                 # Note: compute_general_metrics_batch expects [B,...], we pass [N_total, C]
                 # Let's compute directly here for simplicity
                 diff = full_preds - full_targets
                 mse = torch.mean(diff ** 2).item()
                 mae = torch.mean(torch.abs(diff)).item()
                 max_ae = torch.max(torch.abs(diff)).item()

                 norm_diff_l2 = torch.linalg.norm(diff)
                 norm_gtr_l2 = torch.linalg.norm(full_targets)
                 rel_l2 = (norm_diff_l2 / (norm_gtr_l2 + EPSILON)).item() * 100.0

                 norm_diff_l1 = torch.linalg.norm(diff, ord=1)
                 norm_gtr_l1 = torch.linalg.norm(full_targets, ord=1)
                 rel_l1 = (norm_diff_l1 / (norm_gtr_l1 + EPSILON)).item() * 100.0

                 final_metrics_dict = {
                    'MSE': mse, 'MAE': mae, 'Max AE': max_ae,
                    'Rel L2 Error (%)': rel_l2, 'Rel L1 Error (%)': rel_l1
                 }

                 # Store and print as before
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
            if plot_coords is not None:
                try:
                    channel_to_plot = 0
                    if plot_prd.shape[-1] <= channel_to_plot: channel_to_plot = 0

                    gtr_np = plot_gtr[:, channel_to_plot]
                    prd_np = plot_prd[:, channel_to_plot]

                    try: # Get variable name
                        var_name_idx = self.metadata.active_variables[channel_to_plot]
                        var_name = self.metadata.names['u'][var_name_idx]
                    except (IndexError, KeyError, TypeError, AttributeError):
                        var_name = f"Channel {channel_to_plot}"

                    plot_save_path = os.path.join(self.path_config.result_path, f"test_sample_0_channel_{channel_to_plot}.png")
                    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

                    plot_3d_comparison_matplotlib(
                        coords=plot_coords, u_gtr=gtr_np, u_prd=prd_np,
                        save_path=plot_save_path, variable_name=var_name,
                        point_size=2.0, view_angle=(25, -135) # Example angle
                    )
                    print(f"Saved plot for first test sample to {plot_save_path}")
                except Exception as plot_err:
                    print(f"Error during 3D plotting of first test sample: {plot_err}")

        elif self.setup_config.distributed:
             pass # Handle non-rank 0 processes