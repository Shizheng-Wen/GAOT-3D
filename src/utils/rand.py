import torch 
import random 
import numpy as np 


def manual_seed(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)