import numpy as np
import torch
import random

def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_random_seed():
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(1)
    np.random.seed(1)

def save_ckpt(path, **kwargs):
    """
        Save checkpoint to the path

        Usage:
        >>> save_ckpt("model/poisson_1000.pt", model=model, optimizer=optimizer, scheduler=scheduler)

        Parameters:
        -----------
            path: str
                path to save the checkpoint
            kwargs: dict
                key: str
                    name of the model
                value: StateFul torch object which has the .state_dict() method
                    save object
        
    """
    for k, v in kwargs.items():
        # Examine whether we need to wrap the model
        if isinstance(v, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            kwargs[k] = v.state_dict()  # save the wrapped model, includding the 'module.' prefix
        else:
            kwargs[k] = v.state_dict()
    torch.save(kwargs, path)

def load_ckpt(path, **kwargs):
    """
        Load checkpoint from the path

        Usage:
        >>> model, optimizer, scheduler = load_ckpt("model/poisson_1000.pt", model=model, optimizer=optimizer, scheduler=scheduler)

        Parameters:
        -----------
            path: str
                path to load the checkpoint
            kwargs: dict
                key: str
                    name of the model
                value: StateFul torch object which has the .state_dict() method
                    save object
        Returns:
        --------
            list of torch object
            [model, optimizer, scheduler]
    """
    ckpt = torch.load(path)

    for k, v in kwargs.items():
        state_dict = ckpt[k]
        model_keys = v.state_dict().keys()
        ckpt_keys = state_dict.keys()

        if all(key.startswith('module.') for key in ckpt_keys) and not any(key.startswith('module.') for key in model_keys):
            new_state_dict = {}
            for key in ckpt_keys:
                new_key = key.replace('module.', '', 1)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict
        elif not any(key.startswith('module.') for key in ckpt_keys) and all(key.startswith('module.') for key in model_keys):
            new_state_dict = {}
            for key in ckpt_keys:
                new_key = 'module.' + key
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        v.load_state_dict(state_dict, strict=False)
    return [i for i in kwargs.values()]
