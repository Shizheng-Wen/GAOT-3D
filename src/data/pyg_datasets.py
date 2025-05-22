import torch
import os
import glob
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import numpy as np


class EnrichedData(Data):
    """Custom Data class to handle increments for bipartite edge indices."""

    def __inc__(self, key, value, *args, **kwargs):
        """
        Specifies how attributes should be incremented when batching.
        key: The name of the attribute.
        value: The attribute's tensor value.
        """
        if key.startswith('encoder_edge_index'):
            # Encoder: edge_index[0] indexes LATENT, edge_index[1] indexes PHYSICAL
            # Increment row 0 by num_latent_nodes, row 1 by num_physical_nodes
            # Ensure 'num_latent_nodes' attribute exists in the Data object!
            return torch.tensor([[self.num_latent_nodes], [self.num_nodes]])
        elif key.startswith('decoder_edge_index'):
            # Decoder: edge_index[0] indexes PHYSICAL, edge_index[1] indexes LATENT
            # Increment row 0 by num_physical_nodes, row 1 by num_latent_nodes
            return torch.tensor([[self.num_nodes], [self.num_latent_nodes]])
        elif key.startswith('encoder_query_counts') or key.startswith('decoder_query_counts'):
             # Counts should not be incremented during batching
             return torch.tensor([0] * value.dim(), dtype=torch.long) # Or return 0 for scalar? Check PyG docs if needed. Assuming tensor counts.
        else:
            # Default PyG behavior for other attributes (like standard edge_index, face, etc.)
            return super().__inc__(key, value, *args, **kwargs)

class VTKMeshDataset(Dataset):
    """
    PyTorch Geometric Dataset for loading preprocessed VTK mesh data.
    Assumes data is preprocessed and saved as individual .pt files containing Data objects.
    """
    def __init__(self, root, order_file, dataset_config, split='train', transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Root directory where the dataset should be saved/found.
                        Contains raw (optional) and processed directories.
            order_file (str): Path to the order.txt file.
            dataset_config (DatasetConfig): Configuration object with train/val/test sizes.
            split (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Data transformation function applied after loading.
            pre_transform (callable, optional): Data transformation function applied before saving processed data.
            pre_filter (callable, optional): Data filtering function applied before saving.
        """
        self.order_file = order_file
        self.dataset_config = dataset_config
        self.split = split
        # Assuming processed files are stored in root/processed/
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load indices after processing ensures processed files are available
        self._load_split_indices()

    @property
    def raw_file_names(self):
        # List raw files if needed for processing (e.g., original VTK paths)
        # For simplicity, assume processing happens externally based on order_file
        return [] # Or return list based on order_file if process() needs it

    @property
    def processed_dir(self) -> str:
        """Returns the path to the directory containing processed pyg files."""
        return os.path.join(self.root, self.dataset_config.processed_folder)
        
    @property
    def processed_file_names(self):
        # Check if processed files exist based on order_file
        # This is used by PyG to check if processing is needed
        with open(self.order_file, 'r') as f:
            order_list = [line.strip() for line in f if line.strip()]
        # Use max_num from config if needed
        # max_num = getattr(self.dataset_config, 'max_num', None) # Example if max_num is added
        # if max_num:
        #      order_list = order_list[:max_num]
        return [f"{fname}.pt" for fname in order_list]

    def download(self):
        # No download necessary if data is local
        pass

    def process(self):
        # This method is called if processed files don't exist.
        # Ideally, the external script already created the .pt files.
        # If not, you could implement the VTK -> Data object conversion here.
        print("Processing raw data (should ideally be done by external script)...")
        # Placeholder: Add the conversion logic from your script here if needed.
        # Ensure it saves files named according to processed_file_names in self.processed_dir
        pass

    def _load_split_indices(self):
        with open(self.order_file, 'r') as f:
            all_filenames = [line.strip() for line in f if line.strip()]

        total_samples = len(all_filenames)
        train_size = self.dataset_config.train_size
        val_size = self.dataset_config.val_size
        test_size = self.dataset_config.test_size 

        # Generate indices based on dataset size and split
        indices = np.arange(total_samples)
        if getattr(self.dataset_config, 'rand_dataset', False):
             rng = np.random.default_rng(seed=42) 
             rng.shuffle(indices)

        if self.split == 'train':
            split_indices = indices[:train_size]
        elif self.split == 'val':
            split_indices = indices[train_size : train_size + val_size]
        elif self.split == 'test':
            split_indices = indices[-test_size :] 
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.split_filenames = [f"{all_filenames[i]}.pt" for i in split_indices]
        print(f"Loaded {len(self.split_filenames)} samples for split '{self.split}'.")

    def len(self):
        return len(self.split_filenames)

    def get(self, idx):
        filepath = os.path.join(self.processed_dir, self.split_filenames[idx])
        try:
            data = torch.load(filepath)
            # Apply normalization here if stats are available and not done in preprocessing
            # Example:
            # if hasattr(self, 'mean') and hasattr(self, 'std'):
            #    data.x = (data.x - self.mean) / (self.std + EPSILON)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed file not found: {filepath}. Ensure preprocessing script was run.")
        except Exception as e:
            print(f"Error loading data for index {idx} (file: {filepath}): {e}")
            raise e 