import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from typing import List

def anim_row(titles:List[str], 
             values:List[torch.Tensor], 
             points:torch.Tensor,
             dt:float, 
             output_path:str = "outputs/animation.mp4", 
             density:int=16):
    points = points.detach().cpu().numpy()
    ncols = len(titles)
    assert len(values) == ncols, f"Expected {ncols} values, but got {len(values)}"
    values = [v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for v in values]
    fig, axes = plt.subplots(ncols=ncols, figsize=(5*ncols, 5))
    imgs = []
    cbs  = []
    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        XX, YY = np.mgrid[0:1:density*1j, 0:1:density*1j]
        z = griddata(points, values[i][0], (XX, YY), method="linear")
        img = ax.matshow(z.reshape(density,density), cmap="jet", interpolation="bilinear", vmin=values[i].min(), vmax=values[i].max())
        cbar = fig.colorbar(img, ax=ax)
        imgs.append(img)
        cbs.append(cbar)
    fig.suptitle(f"Time: 0.0s")
    
    def update(frame):
        for i, img in enumerate(imgs):
            z = griddata(points, values[i][frame], (XX, YY), method="linear")
            img.set_data(z.reshape(density,density))
        fig.suptitle(f"Time: {frame*dt:.2f}s")

    anim = FuncAnimation(fig, update, frames=range(values[0].shape[0]), interval=100)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anim.save(output_path)

