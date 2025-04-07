import os 
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from .base import TrainerBase
from .utils.metric import compute_batch_errors, compute_final_metric
from .utils.plot import plot_3d_comparison_pyvista, plot_3d_comparison_matplotlib
from .utils.data_pairs import CustomDataset


from src.data.dataset import Metadata, DATASET_METADATA
from src.model import init_model
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
   
    def init_dataset(self, dataset_config):
        self.latent_token_size = self.model_config.args.latent_tokens
        base_path = dataset_config.base_path
        dataset_name = dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")

        with xr.open_dataset(dataset_path) as ds:
            u_array = ds[self.metadata.group_u].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            else:
                c_array = None
            if self.metadata.group_x is not None:
                x_array = ds[self.metadata.group_x].values # Shape: [num_samples, num_timesteps, num_nodes, num_dims]
            
    
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        self.num_input_channels = x_array.shape[-1]
        self.num_output_channels = u_array.shape[-1]
        
        # Compute dataset sizes
        total_samples = u_array.shape[0]
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        test_size = dataset_config.test_size
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
        assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."
    
        if dataset_config.rand_dataset:
            indices = np.random.permutation(len(u_array))
        else:
            indices = np.arange(len(u_array))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[-test_size:]

        # Split data into train, val, test
        u_train = u_array[train_indices]
        u_val = u_array[val_indices]
        u_test = u_array[test_indices]
        x_train = x_array[train_indices]
        x_val = x_array[val_indices]
        x_test = x_array[test_indices]
        if c_array is not None:
            c_train = c_array[train_indices]
            c_val = c_array[val_indices]
            c_test = c_array[test_indices]
        else:
            c_train = c_val = c_test = None
        
        # Compute dataset statistics from training data
        # Reshape u_train to [num_samples * num_timesteps * num_nodes, num_active_vars]
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = np.mean(u_train_flat, axis=0)
        u_std = np.std(u_train_flat, axis=0) + EPSILON  # Avoid division by zero

        # Store statistics as torch tensors
        self.u_mean = torch.tensor(u_mean, dtype=self.dtype)
        self.u_std = torch.tensor(u_std, dtype=self.dtype)

        # Normalize data using NumPy operations
        u_train = (u_train - u_mean) / u_std
        u_val = (u_val - u_mean) / u_std
        u_test = (u_test - u_mean) / u_std

        # If c is used, compute statistics and normalize c
        if c_array is not None:
            c_train_flat = c_train.reshape(-1, c_train.shape[-1])
            c_mean = np.mean(c_train_flat, axis=0)
            c_std = np.std(c_train_flat, axis=0) + EPSILON  # Avoid division by zero

            # Store statistics
            self.c_mean = torch.tensor(c_mean, dtype=self.dtype)
            self.c_std = torch.tensor(c_std, dtype=self.dtype)

            # Normalize c
            c_train = (c_train - c_mean) / c_std
            c_val = (c_val - c_mean) / c_std
            c_test = (c_test - c_mean) / c_std

        u_train, x_train = torch.tensor(u_train, dtype=self.dtype), torch.tensor(x_train, dtype=self.dtype)
        u_val, x_val = torch.tensor(u_val, dtype=self.dtype), torch.tensor(x_val, dtype=self.dtype)
        u_test, x_test = torch.tensor(u_test, dtype=self.dtype), torch.tensor(x_test, dtype=self.dtype)

        print("Starting Graph Build ...")
        graph_start_time = time.time()
        nb_search = NeighborSearch(
            use_torch_cluster=self.model_config.args.magno.gno_use_torch_cluster)
        gno_radius = self.model_config.args.magno.gno_radius
        scales = self.model_config.args.magno.scales
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

        def custom_collate_fn(batch):
            inputs = torch.stack([item[0] for item in batch])          
            labels = torch.stack([item[1] for item in batch])         
            coords = torch.stack([item[2] for item in batch])          
            encoder_graphs = [item[3] for item in batch] 
            decoder_graphs = [item[4] for item in batch]  
            return inputs, labels, coords, encoder_graphs, decoder_graphs
        
        if self.setup_config.train:
            print("Starting Training Graph Build ...")
            encoder_graphs_train = []
            decoder_graphs_train = []
            for i in range(len(x_train)):
                x_coord = rescale(x_train[i, 0], (-1, 1))
                encoder_nbrs = []
                for scale in scales:
                    scaled_radius = gno_radius * scale
                    with torch.no_grad():
                        nbrs = nb_search(
                            data = x_coord, 
                            queries = self.latent_tokens, 
                            radi = scaled_radius,
                            device = self.device)
                    encoder_nbrs.append(nbrs)
                encoder_graphs_train.append(encoder_nbrs)

                decoder_nbrs = []
                for scale in scales:
                    scaled_radius = gno_radius * scale
                    with torch.no_grad():
                        nbrs = nb_search(
                            data = self.latent_tokens, 
                            queries = x_coord, 
                            radi = scaled_radius,
                            device = self.device)
                    decoder_nbrs.append(nbrs)
                decoder_graphs_train.append(decoder_nbrs)

            train_ds = CustomDataset(
                inputs = x_train,
                labels = u_train, 
                coords = x_train,
                encoder_graphs = encoder_graphs_train,
                decoder_graphs = decoder_graphs_train)
            
            train_sampler = None
            if self.setup_config.distributed:
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=self.setup_config.world_size,
                    rank=self.setup_config.local_rank
                )

            self.train_loader = DataLoader(
                train_ds,
                batch_size = dataset_config.batch_size,
                shuffle = (train_sampler is None and dataset_config.shuffle),
                collate_fn = custom_collate_fn,
                num_workers = dataset_config.num_workers,
                sampler=train_sampler
            )

            encoder_graphs_val = []
            decoder_graphs_val = []
            for i in range(len(x_val)):
                x_coord = rescale(x_val[i, 0], (-1, 1))
                encoder_nbrs = []
                for scale in scales:
                    scaled_radius = gno_radius * scale
                    with torch.no_grad():
                        nbrs = nb_search(
                            data = x_coord, 
                            queries = self.latent_tokens, 
                            radi = scaled_radius,
                            device = self.device)
                    encoder_nbrs.append(nbrs)
                encoder_graphs_val.append(encoder_nbrs)

                decoder_nbrs = []
                for scale in scales:
                    scaled_radius = gno_radius * scale
                    with torch.no_grad():
                        nbrs = nb_search(
                            data = self.latent_tokens,
                            queries = x_coord, 
                            radi = scaled_radius,
                            device = self.device)
                    decoder_nbrs.append(nbrs)
                decoder_graphs_val.append(decoder_nbrs)
            
            val_ds = CustomDataset(
                inputs = x_val,
                labels = u_val, 
                coords = x_val,
                encoder_graphs = encoder_graphs_val,
                decoder_graphs = decoder_graphs_val)
            
            self.val_loader = DataLoader(
                val_ds,
                batch_size=dataset_config.batch_size, 
                shuffle=False, 
                collate_fn=custom_collate_fn,
                num_workers=dataset_config.num_workers
            )

        print("Starting Testing Graph Build ...")
        encoder_graphs_test = []
        decoder_graphs_test = []
        for i in range(len(x_test)):
            x_coord = rescale(x_test[i, 0], (-1,1))
            encoder_nbrs = []
            for scale in scales:
                scaled_radius = gno_radius * scale
                with torch.no_grad():
                    nbrs = nb_search(
                        data = x_coord, 
                        queries = self.latent_tokens, 
                        radi = scaled_radius,
                        device = self.device)
                encoder_nbrs.append(nbrs)
            encoder_graphs_test.append(encoder_nbrs)


            decoder_nbrs = []
            for scale in scales:
                scaled_radius = gno_radius * scale
                with torch.no_grad():
                    nbrs = nb_search(
                        data = self.latent_tokens, 
                        queries = x_coord, 
                        radi = scaled_radius,
                        device = self.device)
                decoder_nbrs.append(nbrs)
            decoder_graphs_test.append(decoder_nbrs)
        
        print(f"Graph building takes {time.time() - graph_start_time} s!")

        test_ds = CustomDataset(
            inputs = x_test,
            labels = u_test, 
            coords = x_test,
            encoder_graphs = encoder_graphs_test,
            decoder_graphs = decoder_graphs_test)

        self.test_loader = DataLoader(
            test_ds, 
            batch_size=dataset_config.batch_size, 
            shuffle=False, 
            collate_fn=custom_collate_fn,
            num_workers=dataset_config.num_workers)
                                                    
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
        
    def train_step(self, batch):
        x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
        x_batch, y_batch, coord_batch = x_batch.to(self.device), y_batch.to(self.device), coord_batch.to(self.device)
        self.latent_tokens = self.latent_tokens.to(self.device)
        encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
        decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
        x_batch, y_batch, coord_batch = x_batch.squeeze(1), y_batch.squeeze(1), coord_batch.squeeze(1)
        pred = self.model(
            pndata = x_batch,
            xcoord = coord_batch, 
            tokens = self.latent_tokens,
            encoder_nbrs = encoder_graph_batch, 
            decoder_nbrs = decoder_graph_batch)
        return self.loss_fn(pred, y_batch)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch in loader:
                x_batch, y_batch, coord_batch = x_batch.to(self.device), y_batch.to(self.device), coord_batch.to(self.device)
                self.latent_tokens = self.latent_tokens.to(self.device)
                encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
                decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
                x_batch, y_batch, coord_batch = x_batch.squeeze(1), y_batch.squeeze(1), coord_batch.squeeze(1)
                pred = self.model(
                    pndata = x_batch,
                    xcoord = coord_batch, 
                    tokens = self.latent_tokens,
                    encoder_nbrs = encoder_graph_batch, 
                    decoder_nbrs = decoder_graph_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def test(self):
        self.model.eval()
        self.model.to(self.device)
        all_relative_errors = []
        first_batch_plotted = False

        with torch.no_grad():
            for i, (x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch) in enumerate(self.test_loader):
                x_batch, y_batch, coord_batch = x_batch.to(self.device), y_batch.to(self.device), coord_batch.to(self.device) # Shape: [batch_size, num_timesteps, num_nodes, num_channels]
                self.latent_tokens = self.latent_tokens.to(self.device)
                encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
                decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
                x_batch, y_batch, coord_batch = x_batch.squeeze(1), y_batch.squeeze(1), coord_batch.squeeze(1)
                pred = self.model(
                    pndata = x_batch,
                    xcoord = coord_batch, 
                    tokens = self.latent_tokens,
                    encoder_nbrs = encoder_graph_batch, 
                    decoder_nbrs = decoder_graph_batch)
                pred_de_norm = pred * self.u_std.to(self.device) + self.u_mean.to(self.device)
                y_sample_de_norm = y_batch * self.u_std.to(self.device) + self.u_mean.to(self.device)
                relative_errors = compute_batch_errors(y_sample_de_norm, pred_de_norm, self.metadata)
                all_relative_errors.append(relative_errors)

                if i == 0 and not first_batch_plotted:
                    try:
                        # Select the last sample in the batch for visualization
                        vis_idx = -1
                        coords_np = coord_batch[vis_idx].cpu().numpy()

                        # --- Select which channel to visualize (e.g., the first one) ---
                        channel_to_plot = 0
                        if pred_de_norm.shape[-1] <= channel_to_plot:
                           print(f"Warning: Channel {channel_to_plot} requested for plotting, but model output only has {pred_de_norm.shape[-1]} channels. Defaulting to channel 0.")
                           channel_to_plot = 0
                        # -------------------------------------------------------------

                        gtr_np = y_sample_de_norm[vis_idx, :, channel_to_plot].cpu().numpy()
                        prd_np = pred_de_norm[vis_idx, :, channel_to_plot].cpu().numpy()

                        # Get variable name from metadata
                        try:
                            var_name = self.metadata.names['u'][self.metadata.active_variables[channel_to_plot]]
                        except (IndexError, KeyError, TypeError):
                            var_name = f"Channel {channel_to_plot}"
                            print(f"Warning: Could not retrieve variable name for channel {channel_to_plot}. Using default.")


                        # Call the PyVista plotting function
                        plot_3d_comparison_matplotlib(
                            coords=coords_np,
                            u_gtr=gtr_np,
                            u_prd=prd_np,
                            save_path=self.path_config.result_path, # Use configured path
                            variable_name=var_name,
                            point_size= 2.0,
                            view_angle= (20, -120)
                            # Optional: adjust point_size, cmap, dpi, view_angle here if needed
                        )
                        first_batch_plotted = True
                    except Exception as plot_err:
                        print(f"Error during 3D plotting of first batch sample: {plot_err}")

        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)

        if self.setup_config.rank == 0:
            self.config.datarow["relative error (direct)"] = final_metric
            print(f"relative error: {final_metric}")
