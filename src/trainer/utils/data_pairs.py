import numpy as np
from ...data.dataset import Metadata
from torch.utils.data import Dataset
import torch
import random

class DynamicPairDataset(Dataset):
    def __init__(self, u_data, c_data, t_values, metadata, max_time_diff=14, 
                 stepper_mode="output", stats=None, use_time_norm=True, dataset_name = None):
        """
        Custom Dataset that generates specific time pairs for training.

        Args:
            u_data (numpy.ndarray): Shape [num_samples, num_timesteps, num_nodes, num_vars]
            c_data (numpy.ndarray): Shape: [num_samples, num_timesteps, num_nodes, num_c_vars] or None
            t_values (numpy.ndarray): Actual time values corresponding to timesteps
            metadata (Metadata): Metadata object containing domain information
            max_time_diff (int): Maximum allowed time difference between t_out and t_in
            stepper_mode (str): Stepper mode for the model [output, residual, time_der]
            stats (dict): Dictionary to store statistics for all variables
            use_time_norm (bool): Normalize time features or not
        """
        self.dataset_name = dataset_name
        self.u_data = u_data
        self.c_data = c_data
        self.t_values = t_values  # Shape: [num_timesteps]
        self.metadata = metadata
        self.stepper_mode = stepper_mode
        self.stats = stats
        self.num_samples, self.num_timesteps, self.num_nodes, self.num_vars = u_data.shape

        # if max_time_diff = 14, Only use the first 15 time steps
        self.num_timesteps = max_time_diff
        self.t_values = self.t_values[:self.num_timesteps + 1] # Shape: [num_timesteps + 1]

        # Generate specific time pairs using index
        self.t_in_indices = []
        self.t_out_indices = []
        for lag in range(2, self.num_timesteps + 1, 2):  # Even lags from 2 to 14
            num_pairs = (self.num_timesteps - lag) // 2 + 1
            for i in range(0, self.num_timesteps - lag + 1, 2):
                t_in_idx = i
                t_out_idx = i + lag
                self.t_in_indices.append(t_in_idx)
                self.t_out_indices.append(t_out_idx)

        self.t_in_indices = np.array(self.t_in_indices)
        self.t_out_indices = np.array(self.t_out_indices)

        self.time_diffs = self.t_values[self.t_out_indices] - self.t_values[self.t_in_indices] # Shape: [num_time_pairs]
        self.num_time_pairs = len(self.t_in_indices)
        self.total_pairs = self.num_samples * self.num_time_pairs # The total size of the training data pairs.

        self.start_times = self.t_values[self.t_in_indices] # Shape: [num_time_pairs]
        
        # precompute the normalized time features for all time pairs
        self.start_times_norm = (self.start_times - self.stats["start_time"]["mean"]) / self.stats["start_time"]["std"]
        self.time_diffs_norm = (self.time_diffs - self.stats["time_diffs"]["mean"]) / self.stats["time_diffs"]["std"]

        self.start_time_expanded = np.tile(self.start_times_norm[:, np.newaxis], (1, self.num_nodes)) # Shape: [num_time_pairs, num_nodes]
        self.time_diff_expanded = np.tile(self.time_diffs_norm[:, np.newaxis], (1, self.num_nodes)) # Shape: [num_time_pairs, num_nodes]

        # Reshape to [num_time_pairs, num_nodes, 1]
        self.start_time_expanded = self.start_time_expanded[..., np.newaxis]
        self.time_diff_expanded = self.time_diff_expanded[..., np.newaxis]

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        # Compute sample index and time pair index
        sample_idx = idx // self.num_time_pairs
        time_pair_idx = idx % self.num_time_pairs

        t_in_idx = self.t_in_indices[time_pair_idx]
        t_out_idx = self.t_out_indices[time_pair_idx]

        # Fetch data for the given indices
        u_in = self.u_data[sample_idx, t_in_idx]  # Input at t_in, Shape: [num_nodes, num_vars]
        u_in_norm = (u_in - self.stats["u"]["mean"]) / self.stats["u"]["std"]
        u_out = self.u_data[sample_idx, t_out_idx]  # Output at t_out, Shape: [num_nodes, num_vars]

        # If c_data is available
        if self.c_data is not None:
            c_in = self.c_data[sample_idx, t_in_idx]
            c_in_norm = (c_in - self.stats["c"]["mean"]) / self.stats["c"]["std"]
        else:
            c_in = None

        # Prepare input features
        input_features = [u_in_norm]
        if c_in is not None:
            input_features.append(c_in_norm)
        input_features = np.concatenate(input_features, axis=-1)

        # Fetch time features
        start_time_expanded = self.start_time_expanded[time_pair_idx] # Shape: [num_nodes, 1]
        time_diff_expanded = self.time_diff_expanded[time_pair_idx] # Shape: [num_nodes, 1]
        # Add normalized time features (expanded to match num_nodes)
        input_features = np.concatenate([input_features, start_time_expanded, time_diff_expanded], axis=-1)

        if self.stepper_mode == "output":
            target = (u_out - self.stats["u"]["mean"]) / self.stats["u"]["std"]
        elif self.stepper_mode == "residual":
            target = ((u_out - u_in) - self.stats["res"]["mean"]) / self.stats["res"]["std"]
        elif self.stepper_mode == "time_der":
            time_diff = self.time_diffs[time_pair_idx]
            target = ((u_out - u_in) / time_diff - self.stats["der"]["mean"]) / self.stats["der"]["std"]
        else:
            raise ValueError(f"Invalid stepper mode: {self.stepper_mode}")

        return input_features, target

class TestDataset(Dataset):
    def __init__(self, u_data, c_data, t_values, metadata, time_indices, stats):
        """
        Custom dataset for testing, providing initial input and ground truth sequences.
        
        Args:
            u_data (numpy.ndarray): Shape [num_samples, num_timesteps, num_nodes, num_vars]
            c_data (numpy.ndarray or None): Shape [num_samples, num_timesteps, num_nodes, num_c_vars]
            t_values (numpy.ndarray): Actual time values corresponding to timesteps
            metadata (Metadata): Metadata object containing domain information
            time_indices (list or np.ndarray): Time indices to consider (e.g., [0, 2, 4, ..., 14])
            stats (dict): 
        """
        self.u_data = u_data
        self.c_data = c_data
        self.t_values = t_values
        self.metadata = metadata
        self.time_indices = time_indices
        self.num_samples = u_data.shape[0]
        self.num_nodes = u_data.shape[2]
        self.num_vars = u_data.shape[3]
        self.dtype = np.float32  # or np.float64, depending on your data type
        self.stats = stats
        # Precompute normalized u_data if necessary (using self.u_mean and self.u_std)
        # Assuming self.u_mean and self.u_std are already computed and available

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get initial input at time t=0
        u_in = self.u_data[idx, self.time_indices[0]]  # Shape: [num_nodes, num_vars]
        u_in_norm = (u_in - self.stats["u"]["mean"]) / self.stats["u"]["std"]
        # Get ground truth outputs at future time steps
        y_sequence = self.u_data[idx, self.time_indices[1:]]  # Shape: [num_timesteps - 1, num_nodes, num_vars]
        
        # If c_data is available
        if self.c_data is not None:
            c_in = self.c_data[idx, self.time_indices[0]]  # Shape: [num_nodes, num_c_vars]
            c_in_norm = (c_in - self.stats["c"]["mean"]) / self.stats["c"]["std"]
            # Combine u_in and c_in
            input_features = np.concatenate([u_in_norm, c_in_norm], axis=-1)
        else:
            input_features = u_in_norm  # Shape: [num_nodes, num_vars]
        
        # Note: Time features will be added in the `autoregressive_predict` function
        return input_features.astype(self.dtype), y_sequence.astype(self.dtype)
    

#######################################
# Utils for Unstructured Graph
#######################################
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, coords, encoder_graphs, decoder_graphs, is_training = True, drop_ratio = 0.3):
        self.inputs = inputs
        self.labels = labels
        self.coords = coords
        self.encoder_graphs = encoder_graphs
        self.decoder_graphs = decoder_graphs
        self.is_training = is_training
        self.drop_ratio = drop_ratio
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        encoder_nbrs = self.encoder_graphs[idx]
        decoder_nbrs = self.decoder_graphs[idx]

        fixed_seed = not self.is_training
        # TODO needs to add the random in the GNO
        # sampled_encoder_nbrs = self.sample_neighbors(encoder_nbrs, fixed_seed=fixed_seed)
        # sampled_decoder_nbrs = self.sample_neighbors(decoder_nbrs, fixed_seed=fixed_seed)

        return self.inputs[idx], self.labels[idx], self.coords[idx], encoder_nbrs, decoder_nbrs

    def sample_neighbors(self, nbrs, fixed_seed=False):
        if fixed_seed:
            torch.manual_seed(0)
        
        sampled_nbrs = []
        for scale_nbr in nbrs:
            neighbors_index = scale_nbr['neighbors_index']
            neighbors_row_splits = scale_nbr['neighbors_row_splits']

            num_edges = neighbors_index.shape[0]
            random_vals = torch.rand(num_edges, device=neighbors_index.device)
            keep_mask = random_vals > self.drop_ratio
            new_neighbors_index = neighbors_index[keep_mask]

            num_nodes = neighbors_row_splits.size(0) - 1
            edge_indices = torch.arange(num_edges, device=neighbors_index.device)
            segments = torch.searchsorted(neighbors_row_splits, edge_indices, right=True) - 1

            keep_mask_int = keep_mask.to(torch.int32)
            new_counts = torch.zeros(num_nodes, dtype=torch.int32, device=neighbors_index.device)
            new_counts = new_counts.scatter_add(0, segments, keep_mask_int)
            new_neighbors_row_splits = torch.cat((torch.tensor([0], dtype=torch.int32, device=neighbors_index.device), new_counts.cumsum(0)))

            sampled_nbrs.append({
                'neighbors_index': new_neighbors_index,
                'neighbors_row_splits': new_neighbors_row_splits  
            })
        return sampled_nbrs
        

#######################################
# Utils for Foundation Model
#######################################
class MixedDataset(Dataset):
    def __init__(self, datasets):
        """
        Custom dataset to handle multiple datasets. 
        Args:
            datasets (dict): Dictionary containing dataset names and corresponding datasets
        """
        self.datasets = datasets
        self.total_length = sum(len(d) for d in datasets.values())
        self.dataset_indices = []
        for dataset_name, dataset in datasets.items():
            for idx in range(len(dataset)):
                self.dataset_indices.append((dataset_name, idx))

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        datasetname, sample_idx = self.dataset_indices[idx]
        dataset = self.datasets[datasetname]
        input_features, target = dataset[sample_idx]
        return datasetname, input_features, target

class CombinedDataLoader:
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        self.loaders_iters = [iter(loader) for loader in self.loaders]
        self.loader_indices = list(range(len(self.loaders_iters)))
        self.current_idx = 0
        return self

    def __next__(self):
        if not self.loader_indices:
            raise StopIteration
        while self.loader_indices:
            idx = self.loader_indices[self.current_idx % len(self.loader_indices)]
            try:
                batch = next(self.loaders_iters[idx])
                self.current_idx += 1
                return batch
            except StopIteration:
                self.loader_indices.remove(idx)
                if not self.loader_indices:
                    raise StopIteration
                self.current_idx = self.current_idx % len(self.loader_indices)
                continue
        raise StopIteration

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)