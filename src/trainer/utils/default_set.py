from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Literal

# model default config
from ...model.layers.attn import TransformerConfig
from ...model.layers.magno import MAGNOConfig

from ..optimizers import OptimizerargsConfig

from omegaconf import OmegaConf

def merge_config(default_config_class, user_config):
    default_config_struct = OmegaConf.structured(default_config_class)
    merged_config = OmegaConf.merge(default_config_struct, user_config)
    return OmegaConf.to_object(merged_config)

@dataclass
class SetUpConfig:
    seed: int = 42                                                      # Random seed for reproducibility
    device: str = "cuda:0"                                              # Computation device, (e.g., "cuda:0", "cpu")   
    dtype: str = "torch.float32"                                        # Data type for computation (e.g., "torch.float32", "torch.float64") 
    trainer_name: str = "static3d"                                      # Type of trainer to use, support [static3d]
    train: bool = True                                                  # Whether to run the training phase  
    test: bool = False                                                  # Whether to run the testing phase
    ckpt: bool = False                                                  # Whether to load the checkpoint
    use_variance_test: bool = False                                     # TODO Whether to use variance testing 
    # Parameters for distributed mode
    distributed: bool = False                                           # Whether to enable distributed training 
    world_size: int = 1                                                 # Total number of processes in distributed training
    rank: int = 0                                                       # Rank of the current process in distributed training
    local_rank: int = 0                                                 # Local rank of the current process in distributed training
    backend: str = "nccl"                                               # Backend for distributed training, e.g., 'nccl' (NVIDIA Collective Communications Library)

@dataclass
class ModelArgsConfig:
    latent_tokens: Tuple[int, int, int] = (64, 64, 64)                         # Size (D, H, W) of latent tokens
    magno: MAGNOConfig = field(default_factory=MAGNOConfig)                    # Configuration for MAGNO model
    transformer: TransformerConfig = field(default_factory=TransformerConfig)  # Configuration for Transformer model

@dataclass
class ModelConfig:
    name: str = "gaot_3d"                                                # Name of the model, support ['gaot_3d']
    use_conditional_norm: bool = False                                   # Whether to use time-conditional normalization (not supported in this repo)
    args: ModelArgsConfig = field(default_factory=ModelArgsConfig)       # Configuration for the model's components

@dataclass
class DatasetConfig:
    name: str = "drivaernet_fullpressure"                                  # Name of the dataset
    metaname: str = "gaot-unstructured/drivaernet_pressure"                # Metaname (identifier) for the dataset, used for loading the dataset
    base_path: str = "/cluster/work/math/camlab-data/graphnpde/drivaernet/"# Base path where the dataset is stored
    processed_folder: str = "processed_pyg"                                # Folder name of the saved/processed .pt files
    use_metadata_stats: bool = False                                       # Whether to use metadata statistics
    sample_rate: float = 0.1                                               # Sampling rate of the dataset
    train_size: int = 5817                                                 # Number of training samples
    val_size: int = 1148                                                   # Number of validation samples
    test_size: int = 1154                                                  # Number of test samples
    rand_dataset: bool = False                                             # Whether to randomize the sequence of loaded dataset
    batch_size: int = 64                                                   # Batch size for training
    num_workers: int = 4                                                   # Number of workers for loading the dataset
    shuffle: bool = True                                                   # Whether to shuffle the dataset                                     
    metric_suite: str = "drivaernet"                                       # Literal["poseidon", "general"]
    # for graph building
    update_pt_files_with_edges: bool = False                               # Flag to trigger edge computation/saving
    use_rescale_new: bool = False                                          # Whether to use new rescale methods for coordinates

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    args: OptimizerargsConfig = field(default_factory=OptimizerargsConfig)

@dataclass
class PathConfig:
    ckpt_path: str = ".ckpt/test/test.pt"                                  # Path to save the checkpoint
    loss_path: str = ".loss/test/test.png"                                  # Path to save the loss plot
    result_path: str = ".result/test/test.png"                              # Path to save the result plot
    database_path: str = ".database/test/test.csv"                          # Path to save the database


