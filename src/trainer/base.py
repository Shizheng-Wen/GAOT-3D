import logging
import os
import dataclasses
from typing import Dict, Any, Optional
import wandb
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

from .optimizers import AdamOptimizer, AdamWOptimizer
from .utils.setup import manual_seed, load_ckpt, save_ckpt
from .utils.default_set import SetUpConfig, ModelConfig, DatasetConfig, OptimizerConfig, PathConfig, merge_config
from ..data.metadata import DATASET_METADATA

class TrainerBase:
    """
    Base class for all trainers, define the init_dataset, initi_model, 
    init_optimizer, train_step, validate, test for coreresponding trainers.

    Attributes:
    ----------
    """
    def __init__(self, args):
        # Config setup
        self.config = args
        self.setup_config = merge_config(SetUpConfig, self.config.setup)
        self.model_config = merge_config(ModelConfig, self.config.model)
        self.dataset_config = merge_config(DatasetConfig, self.config.dataset)
        self.optimizer_config = merge_config(OptimizerConfig, self.config.optimizer)
        self.path_config = merge_config(PathConfig, self.config.path)
        
        self.metadata = DATASET_METADATA[self.dataset_config.metaname]

        # initialization the distributed learning environment
        if self.setup_config.distributed:
            self.init_distributed_mode()
            torch.cuda.set_device(self.setup_config.local_rank)
            self.device = torch.device('cuda', self.setup_config.local_rank)
            logging.getLogger(__name__).info(f'Rank {self.setup_config.rank}: Using device cuda:{self.setup_config.local_rank}')
        else:
            self.device = torch.device(self.setup_config.device)
            logging.getLogger(__name__).info(f'Using device {self.setup_config.device}')
        
        manual_seed(self.setup_config.seed + self.setup_config.rank)

        self._init_wandb()

        if self.setup_config.dtype in ["float", "torch.float32", "torch.FloatTensor"]:
            self.dtype = torch.float32
        elif self.setup_config.dtype in ["double", "torch.float64", "torch.DoubleTensor"]:
            self.dtype = torch.float64
        else:
            raise ValueError(f"Invalid dtype: {self.setup_config.dtype}")
        self.loss_fn = nn.MSELoss()
        
        self.init_dataset(self.dataset_config)
        self.init_model(self.model_config)
        self.init_optimizer(self.optimizer_config)

        if self.setup_config.rank == 0:
            nparam = sum(
                [p.numel() * 2 if p.is_complex() else p.numel() for p in self.model.parameters()]
            )
            nbytes = sum(
                [p.numel() * 2 * p.element_size() if p.is_complex() else p.numel() * p.element_size() for p in self.model.parameters()]
            )
            logging.getLogger(__name__).info(f"Number of parameters: {nparam}")
            args.datarow['nparams'] = nparam
            args.datarow['nbytes'] = nbytes

# ------------ init ------------ #
    def init_dataset(self, dataset_config):
        raise NotImplementedError
    
    def init_model(self, model_config):
        raise NotImplementedError

    def init_optimizer(self, optimizer_config):
        """Initialize the optimizer"""
        self.optimizer = {
            "adam": AdamOptimizer,
            "adamw": AdamWOptimizer
        }[self.optimizer_config.name](self.model.parameters(), self.optimizer_config.args)

    def init_distributed_mode(self):
        logger = logging.getLogger(__name__)
        
        # --- START: MODIFICATIONS FOR SLURM ---
        if 'SLURM_PROCID' in os.environ:
            # Running on Slurm, retrieve settings from Slurm environment variables
            self.setup_config.rank = int(os.environ['SLURM_PROCID'])
            self.setup_config.world_size = int(os.environ['SLURM_NPROCS'])
            self.setup_config.local_rank = int(os.environ['SLURM_LOCALID'])
            
            # PyTorch's init_process_group uses these standard variables, so we set them.
            os.environ['RANK'] = str(self.setup_config.rank)
            os.environ['WORLD_SIZE'] = str(self.setup_config.world_size)
            os.environ['LOCAL_RANK'] = str(self.setup_config.local_rank)
            
            # The master address and port are needed for synchronization.
            # srun typically handles this, but it's robust to set it explicitly.
            # The master is always rank 0.
            host = os.environ['SLURM_STEP_NODELIST'].split(' ')[0]
            os.environ['MASTER_ADDR'] = host
            os.environ['MASTER_PORT'] = '29500' # A standard free port

        elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.setup_config.rank = int(os.environ['RANK'])
            self.setup_config.world_size = int(os.environ['WORLD_SIZE'])
            self.setup_config.local_rank = int(os.environ['LOCAL_RANK'])  # 移除默认值，如果没有设置应该报错
        else:
            logger.info('Not using distributed mode')
            self.setup_config.distributed = False
            self.setup_config.rank = 0
            self.setup_config.world_size = 1
            self.setup_config.local_rank = 0
            return

        logger.info(f'Distributed mode initialized - Rank: {self.setup_config.rank}, '
                   f'World Size: {self.setup_config.world_size}, '
                   f'Local Rank: {self.setup_config.local_rank}')

        dist.init_process_group(
            backend=self.setup_config.backend,
            init_method='env://',
            world_size=self.setup_config.world_size,
            rank=self.setup_config.rank
        )
        dist.barrier()

# ------------ wandb ------------ #

    def _to_serializable_config(self) -> Dict[str, Any]:
        """Convert config objects to a nested dict suitable for logging."""
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            if isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            return obj

        payload: Dict[str, Any] = {}
        payload['setup'] = to_dict(self.setup_config)
        payload['model'] = to_dict(self.model_config)
        payload['optimizer'] = {
            'name': self.optimizer_config.name,
            'args': to_dict(self.optimizer_config.args)
        }
        payload['dataset'] = to_dict(self.dataset_config)
        payload['path'] = to_dict(self.path_config)
        return payload

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run if enabled in setup config."""
        self.wandb_run = None
        logger = logging.getLogger(__name__)
        
        try:
            wandb_enabled = getattr(self.setup_config, 'wandb', False)
            current_rank = self.setup_config.rank
            logger.info(f'Rank {current_rank}: wandb_enabled={wandb_enabled}, checking if should init wandb (only rank 0 should)')
            
            if wandb_enabled and current_rank == 0:
                import os
                import wandb
                logger.info(f'Rank {current_rank}: Initializing wandb...')
                # Respect mode: 'online' | 'offline' | 'disabled'
                if hasattr(self.setup_config, 'wandb_mode') and self.setup_config.wandb_mode:
                    os.environ['WANDB_MODE'] = str(self.setup_config.wandb_mode)

                init_kwargs = {
                    'project': self.setup_config.wandb_project,
                    'entity': self.setup_config.wandb_entity,
                    'name': self.setup_config.wandb_run_name,
                    'group': self.setup_config.wandb_group,
                    'notes': self.setup_config.wandb_notes,
                    'tags': self.setup_config.wandb_tags,
                    'config': self._to_serializable_config(),
                    'reinit': True
                }
                # Remove None values to avoid API complaints
                init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
                self.wandb_run = wandb.init(**init_kwargs)

                if getattr(self.setup_config, 'wandb_watch_model', False):
                    wandb.watch(self.model, log='all', log_freq=100)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to initialize Weights & Biases: {e}")
            self.wandb_run = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B if enabled."""
        if getattr(self, 'wandb_run', None) is None:
            return
        try:
            import wandb
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to log metrics to W&B: {e}")

    def save_checkpoint_artifact(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint as wandb artifact."""

        artifact_name = self.setup_config.wandb_run_name
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model", 
            description=f"Model checkpoint at epoch {epoch}"
        )
        
        
        artifact.add_file(self.path_config.ckpt_path, name=os.path.basename(self.path_config.ckpt_path))
            
            # Add metadata
        artifact.metadata = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "is_best": is_best,
            "model_config": self._to_serializable_config()["model"],
            "dataset_config": self._to_serializable_config().get("dataset", {}),
        }
            
        aliases = ["latest"]
        if is_best:
            aliases.append("best")
                
        wandb.log_artifact(artifact, aliases=aliases)
        logging.getLogger(__name__).info(f"Saved checkpoint artifact: {artifact_name}:{aliases}")

    def load_from_artifact(self, artifact_name: str) -> Optional[Dict[str, Any]]:
        artifact = wandb.use_artifact(artifact_name)
        artifact_dir = os.path.join(".artifacts", artifact.name)
        if not os.path.exists(artifact_dir):
            logging.getLogger(__name__).info(f"Didn't find artifact in local, downloading artifact from wandb: {artifact_name}")
            artifact_dir = artifact.download(root=artifact_dir)
        
        ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if len(ckpt_files) > 1:
            logging.getLogger(__name__).warning(f"Multiple .pt files found, using first one: {ckpt_files[0]}")
        
        ckpt_path = os.path.join(artifact_dir, ckpt_files[0])
        self.load_ckpt(ckpt_path)

        logging.getLogger(__name__).info(f"Successfully loaded model from artifact: {artifact_name}")
        
        return artifact.metadata

    def _finish_wandb(self) -> None:
        if getattr(self, 'wandb_run', None) is None:
            return
        try:
            import wandb
            wandb.finish()
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to finalize W&B run: {e}")
    
# ------------ utils ------------ #
    def to(self, device):
        self.model.to(device)
    
    def type(self, dtype):
        # TODO: check if this is necessary, dataloader does not have type method
        self.model.type(dtype)
        self.train_loader.type(dtype)
        self.val_loader.type(dtype)
        self.test_loader.type(dtype)

    def load_ckpt(self, path: Optional[str] = None):
        if path is None:
            path = self.path_config.ckpt_path
        load_ckpt(path, model=self.model)

        return self
    
    def save_ckpt(self, path: Optional[str] = None):
        """Save checkpoint to the config.ckpt_path"""
        if path is None:
            path = self.path_config.ckpt_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_ckpt(path, model = self.model)

        return self

    def compute_test_errors(self):
        # TODO: compute test errors (need to modulate based on the type of dataset, based on metadata)
        raise NotImplementedError

# ------------ train ------------ #
    def train_step(self, batch):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        x_batch, y_batch = x_batch.squeeze(1), y_batch.squeeze(1)
        pred = self.model(self.rigraph, x_batch)
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_batch, y_batch = x_batch.squeeze(1), y_batch.squeeze(1)
                pred = self.model(self.rigraph, x_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, verbose=False):
        #self.to(self.device)
        #self.type(self.dtype)

        result = self.optimizer.optimize(self)
        self.config.datarow['training time'] = result['time']
    

        if len(result['train']['loss']) == 0:
            if self.setup_config.rank == 0:
                self.test()
        else:
            kwargs = {
                "epochs": result['train']['epoch'],
                "losses": result['train']['loss']
            }
        
            if "valid" in result:
                kwargs['val_epochs'] = result['valid']['epoch']
                kwargs['val_losses'] = result['valid']['loss']
            
            if "best" in result:
                kwargs['best_epoch'] = result['best']['epoch']
                kwargs['best_loss'] = result['best']['loss']
            
            if self.setup_config.rank == 0:
                self.plot_losses(**kwargs)
                self.log_metrics({
                    'best/loss': result['best']['loss'],
                    'best/epoch': result['best']['epoch'],
                    'training_time': result['time']
                })
                self.save_ckpt()
                if self.setup_config.wandb:
                    self.save_checkpoint_artifact(
                        epoch=result['best']['epoch'], 
                        train_loss=result['best']['loss'],
                        val_loss=result['best']['loss'],
                        is_best=True
                        )
                self.test()

        if self.setup_config.rank == 0:
            self._finish_wandb()

# ------------ plot ------------ #
    def plot_losses(self, 
                    epochs,
                    losses, 
                    val_epochs = None,
                    val_losses = None,
                    best_epoch = None,
                    best_loss  = None):
        
        if val_losses is None:
            # plot only train loss
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, losses)
            ax.scatter([best_epoch],[best_loss], c='r', marker='o', label="best loss")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss vs Epoch')
            ax.legend()
            ax.set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax.set_yscale('log')
            np.savez(self.path_config["loss_path"][:-4]+".npz", epochs=epochs, losses=losses)

        else:
            # also plot valid loss
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            ax[0].plot(epochs, losses)
            ax[0].scatter([best_epoch],[best_loss], c='r', marker='o', label="best loss")
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].set_title('Loss vs Epoch')
            ax[0].legend()
            ax[0].set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax[0].set_yscale('log')

            ax[1].plot(val_epochs, val_losses)
            # if best_epoch is not None and best_loss is not None:
            #     ax[1].scatter([best_epoch],[best_loss], c='r', marker='o', label="best validation loss")
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('relative error')
            ax[1].set_title('Loss vs relative error')
            ax[1].legend()
            ax[1].set_xlim(left=0)
            if (np.array(val_losses) > 0).all():
                ax[1].set_yscale('log')
            plt.savefig(self.path_config.loss_path)
            np.savez(self.path_config.loss_path[:-4]+".npz", epochs=epochs, losses=losses, val_epochs=val_epochs, val_losses=val_losses)

    def plot_results(self):
        raise NotImplementedError

# ------------ test ------------ #
    def variance_test(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
        
