from typing import Tuple, Optional, Union
from dataclasses import dataclass
from omegaconf import OmegaConf
import math

from .gaot_3d import GAOT3D

def init_model(input_size:int, 
                output_size:int, 
                model:str,
                config:dataclass = None
                ):
    
    supported_models = [
        "gaot3d"
    ]

    if model.lower() == 'gaot_3d':
        return GAOT3D(
            input_size=input_size,
            output_size=output_size,
            magno_config=config.magno,
            attn_config=config.transformer,
            latent_tokens=config.latent_tokens
        )
    
    else:
        raise ValueError(f"model {model} not supported currently!")