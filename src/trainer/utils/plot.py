import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import matplotlib.tri as tri
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import seaborn as sns
from typing import Optional, Tuple, Union, List
import torch
from sklearn.decomposition import PCA


from ...data.dataset import Metadata
from .cal_metric import compute_batch_errors, compute_final_metric

C_BLACK = '#000000'
C_WHITE = '#ffffff'
C_BLUE = '#093691'
C_RED = '#911b09'
C_BLACK_BLUEISH = '#011745'
C_BLACK_REDDISH = '#380801'
C_WHITE_BLUEISH = '#dce5f5'
C_WHITE_REDDISH = '#f5dcdc'

# bright ones
C_BRIGHT_PURPLE = '#7f00ff'   # 亮紫
C_BRIGHT_PINK   = '#ff00ff'   # 玫红
C_BRIGHT_ORANGE = '#ff7700'   # 橙
C_BRIGHT_YELLOW = '#ffdd00'   # 黄
C_BRIGHT_GREEN  = '#00ee00'   # 亮绿
C_BRIGHT_CYAN   = '#00ffff'   # 青
C_BRIGHT_BLUE   = '#0f00ff'   # 亮蓝


CMAP_BBR = matplotlib.colors.LinearSegmentedColormap.from_list(
  'blue_black_red',
  [C_WHITE_BLUEISH, C_BLUE, C_BLACK, C_RED, C_WHITE_REDDISH],
  N=200,
)
CMAP_BWR = matplotlib.colors.LinearSegmentedColormap.from_list(
  'blue_white_red',
  [C_BLACK_BLUEISH, C_BLUE, C_WHITE, C_RED, C_BLACK_REDDISH],
  N=200,
)
CMAP_WRB = matplotlib.colors.LinearSegmentedColormap.from_list(
  'white_red_black',
  [C_WHITE, C_RED, C_BLACK],
  N=200,
)

CMAP_BRIGHT_SYMM = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bright_symm',
    [
        '#1f00ff',  # 亮蓝
        '#ffffff',  # 白色
        '#ff005f',  # 亮红/粉
    ],
    N=256,
)
CMAP_BRIGHT_SINGLE = matplotlib.colors.LinearSegmentedColormap.from_list(
    'bright_single',
    [
        '#ffe5ee',  # 更柔和的浅粉 (类似 web 色 'MistyRose')
        '#ff77aa',  # 中间的粉红/玫红
        '#ff0050',  # 亮红，与 symm colormap 的红统一
    ],
    N=256,
)

plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.family'] = 'serif'
SCATTER_SETTINGS = dict(marker='s', s=1, alpha=1, linewidth=0)
#HATCH_SETTINGS = dict(facecolor='#b8b8b8', hatch='//////', edgecolor='#4f4f4f', linewidth=.0)
HATCH_SETTINGS = dict(facecolor='#b8b8b8', edgecolor='#4f4f4f', linewidth=.0)

BACKGROUND_SETTINGS_BRIGHT = dict(
    facecolor='#ffe5ee',  # 浅粉，与亮色 colormap 呼应
    alpha=0.3,            # 半透明度，可视情况调整或移除
    edgecolor='none',     # 不绘制边框
    linewidth=0.0,        # 如无需边框，可设为 0
)
def plot_trajectory(u, x, t, idx_t, idx_s=0, symmetric=True, ylabels=None, domain=([0, 0], [1, 1])):

  _WIDTH_PER_COL = 1.5
  _HEIGHT_PER_ROW = 1.7
  _WIDTH_MARGIN = .2
  _HEIGHT_MARGIN = .2
  _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
  _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .42 * _HEIGHT_PER_ROW
  _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x.shape[2] ** .5)

  # Arrange the inputs
  n_vars = u.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  # Create the figure and the gridspec
  figsize=(_WIDTH_PER_COL*len(idx_t)+_WIDTH_MARGIN, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
  fig = plt.figure(figsize=figsize,)
  g = fig.add_gridspec(
    nrows=n_vars,
    ncols=len(idx_t)+1,
    width_ratios=([1]*len(idx_t) + [.1]),
    wspace=0.05,
    hspace=0.20,
  )
  # Add all axes
  axs = []
  for r in range(n_vars):
    row = []
    for c in range(len(idx_t)):
      row.append(fig.add_subplot(g[r, c]))
    axs.append(row)
  axs = np.array(axs)
  # Settings
  for ax in axs.flatten():
    ax: plt.Axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([domain[0][0], domain[1][0]])
    ax.set_ylim([domain[0][1], domain[1][1]])

  # Add hatch to the background
  print([np.min(x[..., 0]), np.max(x[..., 0])])
  for ax in axs.flatten():
    ax.fill_between(
      x=[domain[0][0], domain[1][0]], y1=domain[0][1], y2=domain[1][1],
      **HATCH_SETTINGS,
    )

  # Loop over variables
  for r in range(n_vars):
    # Set cmap and colorbar range
    if symmetric[r]:
      cmap = CMAP_BWR
      vmax = np.max(np.abs(u[idx_s, idx_t, ..., r]))
      vmin = -vmax
    else:
      cmap = CMAP_WRB
      vmax = np.max(u[idx_s, idx_t, ..., r])
      vmin = np.min(u[idx_s, idx_t, ..., r])

    # Loop over columns
    for icol in range(len(idx_t)):
      h = axs[r, icol].scatter(
        x=x[idx_s, idx_t[icol], ..., 0],
        y=x[idx_s, idx_t[icol], ..., 1],
        c=u[idx_s, idx_t[icol], ..., r],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **_SCATTER_SETTINGS,
      )
      if (r == 0) and (len(idx_t) > 1):
        axs[r, icol].set(title=f'$t=t_{{{idx_t[icol]}}}$')

    # Add colorbar
    ax_cb = fig.add_subplot(g[r, -1])
    cb = plt.colorbar(h, cax=ax_cb)
    cb.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.get_offset_text().set(size=8)
    ax_cb.yaxis.set_tick_params(labelsize=8)

  # Add ylabels
  for r in range(n_vars):
    label = ylabels[r] if ylabels else f'Variable {r:02d}'
    axs[r, 0].set(ylabel=label);

  return fig, axs

def plot_estimates(
    u_inp: np.ndarray,
    u_gtr: np.ndarray,
    u_prd: np.ndarray,
    x_inp: np.ndarray,
    x_out: np.ndarray,
    symmetric: Union[bool, List[bool]] = True,
    names: Optional[List[str]] = None,
    domain: Tuple[List[float], List[float]] = ([0, 0], [1, 1]),
) -> plt.Figure:
    """
    Plots input data, ground-truth, model predictions, and absolute errors over a 2D domain.

    This function creates a figure with four panels (columns) for each variable:
    1) Input data,
    2) Ground-truth values,
    3) Model predictions,
    4) Absolute error (|ground-truth - prediction|).

    A horizontal colorbar is provided for each column, showing the data range used for coloring.
    
    Parameters
    ----------
    u_inp : np.ndarray
        The input data array of shape (N_inp, n_input_vars), where:
          - N_inp is the number of input points.
          - n_input_vars is the number of input variables (e.g., different physical quantities).
    u_gtr : np.ndarray
        The ground-truth data array of shape (N_out, n_output_vars). N_out can differ from N_inp
        if the input and output grids do not match.
    u_prd : np.ndarray
        The model-predicted data array, same shape as `u_gtr` (i.e., (N_out, n_output_vars)).
        This is compared against `u_gtr` to compute the absolute error.
    x_inp : np.ndarray
        The (x, y) coordinates of each input point, shape (N_inp, 2).
        Used for the scatter plot of `u_inp`.
    x_out : np.ndarray
        The (x, y) coordinates for the output/ground-truth grid, shape (N_out, 2).
        Used for the scatter plots of `u_gtr`, `u_prd`, and their absolute error.
    symmetric : bool or list of bool, optional
        Whether to use a symmetric color scale (colormap) for each variable. 
        If True, the color limits are set to [-vmax, +vmax], where vmax is 
        the maximum absolute value across data samples for that variable. 
        If a list of booleans is provided, each element corresponds to one variable.
    names : list of str, optional
        A list of variable names (of length n_vars) used as labels on the vertical axis.
        If None, default labels such as "Variable 00", "Variable 01", etc., are used.
    domain : tuple of list, optional
        Defines the displayed plotting region as ([x_min, y_min], [x_max, y_max]).
        Defaults to ([0, 0], [1, 1]). The background hatch pattern will fill this region.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the subplots. Each variable has one row in the figure,
        and there are four columns of scatter plots: input, ground-truth, prediction,
        and absolute error. Each column is accompanied by a horizontal colorbar.

    Notes
    -----
    - Internally, the function arranges subplots for each variable in a two-row layout:
      the top row is for the actual scatter plots, and the bottom row hosts the colorbars.
    - The absolute error is plotted as |u_gtr - u_prd|.
    - Use the returned figure object to further customize, save, or display the figure.

    Examples
    --------
    >>> import numpy as np
    >>> # Assume we have two variables (n_vars = 2)
    >>> # Input grid has 50 points, output grid has 100 points
    >>> x_inp = np.random.rand(50, 2)
    >>> x_out = np.random.rand(100, 2)
    >>> u_inp = np.random.randn(50, 2)
    >>> u_gtr = np.random.randn(100, 2)
    >>> u_prd = u_gtr + 0.1 * np.random.randn(100, 2)
    >>> fig = plot_estimates(
    ...     u_inp=u_inp,
    ...     u_gtr=u_gtr,
    ...     u_prd=u_prd,
    ...     x_inp=x_inp,
    ...     x_out=x_out,
    ...     symmetric=True,
    ...     names=["Temperature", "Concentration"],
    ...     domain=([0, 0], [1, 1])
    ... )
    >>> fig.tight_layout()
    >>> fig.show()
    """
    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = .2
    _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x_inp.shape[0] ** .5)

    n_vars = u_gtr.shape[-1]
    if isinstance(symmetric, bool):
        symmetric = [symmetric] * n_vars

    # Create the figure and the gridspec
    figsize=(8.6, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(
        nrows=n_vars,
        ncols=1,
        wspace=0,
        hspace=0,
    )

    figs = []
    for ivar in range(n_vars):
        figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
    # Add axes
    axs_inp = []
    axs_gtr = []
    axs_prd = []
    axs_err = []
    axs_cb_inp = []
    axs_cb_out = []
    axs_cb_err = []
    for ivar in range(n_vars):
        g = figs[ivar].add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[1, .05],
        wspace=0.20,
        hspace=0.05,
        )
        axs_inp.append(figs[ivar].add_subplot(g[0, 0]))
        axs_gtr.append(figs[ivar].add_subplot(g[0, 1]))
        axs_prd.append(figs[ivar].add_subplot(g[0, 2]))
        axs_err.append(figs[ivar].add_subplot(g[0, 3]))
        axs_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
        axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))
        axs_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
    # Settings
    for ax in [ax for axs in [axs_inp, axs_gtr, axs_prd, axs_err] for ax in axs]:
        ax: plt.Axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([domain[0][0], domain[1][0]])
        ax.set_ylim([domain[0][1], domain[1][1]])
        ax.fill_between(
        x=[domain[0][0], domain[1][0]], y1=domain[0][1], y2=domain[1][1],
        **HATCH_SETTINGS,
        )

    # Get prediction error
    u_err = (u_gtr - u_prd)

    # Loop over variables
    for ivar in range(n_vars):
        # Get ranges
        vmax_inp = np.max(u_inp[:, ivar])
        vmax_gtr = np.max(u_gtr[:, ivar])
        vmax_prd = np.max(u_prd[:, ivar])
        vmax_out = max(vmax_gtr, vmax_prd)
        vmin_inp = np.min(u_inp[:, ivar])
        vmin_gtr = np.min(u_gtr[:, ivar])
        vmin_prd = np.min(u_prd[:, ivar])
        vmin_out = min(vmin_gtr, vmin_prd)
        abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
        abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))

        # Plot input
        h = axs_inp[ivar].scatter(
        x=x_inp[:, 0],
        y=x_inp[:, 1],
        c=u_inp[:, ivar],
        cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
        vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
        vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
        **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_inp[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))
        # Plot ground truth
        h = axs_gtr[ivar].scatter(
        x=x_out[:, 0],
        y=x_out[:, 1],
        c=u_gtr[:, ivar],
        cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
        vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
        vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
        **_SCATTER_SETTINGS,
        )
        # Plot estimate
        h = axs_prd[ivar].scatter(
        x=x_out[:, 0],
        y=x_out[:, 1],
        c=u_prd[:, ivar],
        cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
        vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
        vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
        **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_out[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))

        # Plot error
        h = axs_err[ivar].scatter(
        x=x_out[:, 0],
        y=x_out[:, 1],
        c=np.abs(u_err[:, ivar]),
        cmap=CMAP_WRB,
        vmin=0,
        vmax=np.max(np.abs(u_err[:, ivar])),
        **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_err[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))

    # Set titles
    axs_inp[0].set(title='Input');
    axs_gtr[0].set(title='Ground-truth');
    axs_prd[0].set(title='Model estimate');
    axs_err[0].set(title='Absolute error');

    # Set variable names
    for ivar in range(n_vars):
        label = names[ivar] if names else f'Variable {ivar:02d}'
        axs_inp[ivar].set(ylabel=label);

    # Rotate colorbar tick labels
    for ax in [ax for axs in [axs_cb_inp, axs_cb_out, axs_cb_err] for ax in axs]:
        ax: plt.Axes
        ax.xaxis.get_offset_text().set(size=8)
        ax.xaxis.set_tick_params(labelsize=8)

    return fig


def plot_estimates_bluff(
    u_inp: np.ndarray,
    u_gtr: np.ndarray,
    u_prd: np.ndarray,
    x_inp: np.ndarray,
    x_out: np.ndarray,
    shape_path: str,
    symmetric: Union[bool, List[bool]] = True,
    names: Optional[List[str]] = None,
    domain: Tuple[List[float], List[float]] = ([0, 0], [1, 1]),
) -> plt.Figure:
    """
    Plots input data, ground-truth, model predictions, and absolute errors over a 2D domain.

    This function creates a figure with four panels (columns) for each variable:
    1) Input data,
    2) Ground-truth values,
    3) Model predictions,
    4) Absolute error (|ground-truth - prediction|).

    A horizontal colorbar is provided for each column, showing the data range used for coloring.
    
    Parameters
    ----------
    u_inp : np.ndarray
        The input data array of shape (N_inp, n_input_vars), where:
          - N_inp is the number of input points.
          - n_input_vars is the number of input variables (e.g., different physical quantities).
    u_gtr : np.ndarray
        The ground-truth data array of shape (N_out, n_output_vars). N_out can differ from N_inp
        if the input and output grids do not match.
    u_prd : np.ndarray
        The model-predicted data array, same shape as `u_gtr` (i.e., (N_out, n_output_vars)).
        This is compared against `u_gtr` to compute the absolute error.
    x_inp : np.ndarray
        The (x, y) coordinates of each input point, shape (N_inp, 2).
        Used for the scatter plot of `u_inp`.
    shape: str
        The path for the shape file.
    x_out : np.ndarray
        The (x, y) coordinates for the output/ground-truth grid, shape (N_out, 2).
        Used for the scatter plots of `u_gtr`, `u_prd`, and their absolute error.
    symmetric : bool or list of bool, optional
        Whether to use a symmetric color scale (colormap) for each variable. 
        If True, the color limits are set to [-vmax, +vmax], where vmax is 
        the maximum absolute value across data samples for that variable. 
        If a list of booleans is provided, each element corresponds to one variable.
    names : list of str, optional
        A list of variable names (of length n_vars) used as labels on the vertical axis.
        If None, default labels such as "Variable 00", "Variable 01", etc., are used.
    domain : tuple of list, optional
        Defines the displayed plotting region as ([x_min, y_min], [x_max, y_max]).
        Defaults to ([0, 0], [1, 1]). The background hatch pattern will fill this region.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the subplots. Each variable has one row in the figure,
        and there are four columns of scatter plots: input, ground-truth, prediction,
        and absolute error. Each column is accompanied by a horizontal colorbar.

    Notes
    -----
    - Internally, the function arranges subplots for each variable in a two-row layout:
      the top row is for the actual scatter plots, and the bottom row hosts the colorbars.
    - The absolute error is plotted as |u_gtr - u_prd|.
    - Use the returned figure object to further customize, save, or display the figure.

    Examples
    --------
    >>> import numpy as np
    >>> # Assume we have two variables (n_vars = 2)
    >>> # Input grid has 50 points, output grid has 100 points
    >>> x_inp = np.random.rand(50, 2)
    >>> x_out = np.random.rand(100, 2)
    >>> u_inp = np.random.randn(50, 2)
    >>> u_gtr = np.random.randn(100, 2)
    >>> u_prd = u_gtr + 0.1 * np.random.randn(100, 2)
    >>> fig = plot_estimates(
    ...     u_inp=u_inp,
    ...     u_gtr=u_gtr,
    ...     u_prd=u_prd,
    ...     x_inp=x_inp,
    ...     x_out=x_out,
    ...     symmetric=True,
    ...     names=["Temperature", "Concentration"],
    ...     domain=([0, 0], [1, 1])
    ... )
    >>> fig.tight_layout()
    >>> fig.show()
    """
    with open(shape_path, "r") as file:
      lines = file.readlines()
    
    data = []
    for line in lines:
      try: 
        x, y = map(float, line.split())
        data.append((x, y))
      except ValueError:
        continue
    shape_data = np.array(data)

    triang = tri.Triangulation(x_inp[:, 0], x_inp[:, 1])
    polygon_path = Path(shape_data)
    mask = np.zeros(triang.triangles.shape[0], dtype = bool)
    for i, triangle in enumerate(triang.triangles):
      pts = x_inp[triangle]
      center = np.mean(pts, axis = 0)
      if polygon_path.contains_point(center):
        mask[i] = True
    triang.set_mask(mask)

    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = .2
    _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x_inp.shape[0] ** .5)

    n_vars = u_gtr.shape[-1]
    if isinstance(symmetric, bool):
        symmetric = [symmetric] * n_vars

    # Create the figure and the gridspec
    figsize=(8.6, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(
        nrows=n_vars,
        ncols=1,
        wspace=0,
        hspace=0,
    )

    figs = []
    for ivar in range(n_vars):
        figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
    # Add axes
    axs_inp = []
    axs_gtr = []
    axs_prd = []
    axs_err = []
    axs_cb_inp = []
    axs_cb_out = []
    axs_cb_err = []
    for ivar in range(n_vars):
        g = figs[ivar].add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[1, .05],
        wspace=0.20,
        hspace=0.05,
        )
        axs_inp.append(figs[ivar].add_subplot(g[0, 0]))
        axs_gtr.append(figs[ivar].add_subplot(g[0, 1]))
        axs_prd.append(figs[ivar].add_subplot(g[0, 2]))
        axs_err.append(figs[ivar].add_subplot(g[0, 3]))
        axs_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
        axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))
        axs_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
    # Settings
    for ax in [ax for axs in [axs_inp, axs_gtr, axs_prd, axs_err] for ax in axs]:
        ax: plt.Axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([domain[0][0], domain[1][0]])
        ax.set_ylim([domain[0][1], domain[1][1]])
        ax.fill_between(
        x=[domain[0][0], domain[1][0]], y1=domain[0][1], y2=domain[1][1],
        **HATCH_SETTINGS,
        )

    # Get prediction error
    u_err = (u_gtr - u_prd)

    # Loop over variables
    for ivar in range(n_vars):
        # Get ranges
        vmax_inp = np.max(u_inp[:, ivar])
        vmax_gtr = np.max(u_gtr[:, ivar])
        vmax_prd = np.max(u_prd[:, ivar])
        vmax_out = max(vmax_gtr, vmax_prd)
        vmin_inp = np.min(u_inp[:, ivar])
        vmin_gtr = np.min(u_gtr[:, ivar])
        vmin_prd = np.min(u_prd[:, ivar])
        vmin_out = min(vmin_gtr, vmin_prd)
        abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
        abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))

        # Plot input
        h = axs_inp[ivar].tricontourf(
          triang,
          u_inp[:, ivar],
          levels = 200,
          cmpa = "jet",
          vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
          vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
        )
        cb = plt.colorbar(h, cax=axs_cb_inp[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))
        # Plot ground truth
        h = axs_gtr[ivar].tricontourf(
          triang,
          u_gtr[:, ivar],
          levels= 200,
          cmap = "jet",
          vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
          vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
        )
        # Plot estimate
        h = axs_prd[ivar].tricontourf(
          triang,
          u_prd[:, ivar],
          levels= 200,
          cmap = "jet",
          vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
          vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
        )
        cb = plt.colorbar(h, cax=axs_cb_out[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))

        # Plot error
        h = axs_err[ivar].scatter(
        x=x_out[:, 0],
        y=x_out[:, 1],
        c=np.abs(u_err[:, ivar]),
        cmap=CMAP_WRB,
        vmin=0,
        vmax=np.max(np.abs(u_err[:, ivar])),
        **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_err[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))

    # Set titles
    axs_inp[0].set(title='Input');
    axs_gtr[0].set(title='Ground-truth');
    axs_prd[0].set(title='Model estimate');
    axs_err[0].set(title='Absolute error');

    # Set variable names
    for ivar in range(n_vars):
        label = names[ivar] if names else f'Variable {ivar:02d}'
        axs_inp[ivar].set(ylabel=label);

    # Rotate colorbar tick labels
    for ax in [ax for axs in [axs_cb_inp, axs_cb_out, axs_cb_err] for ax in axs]:
        ax: plt.Axes
        ax.xaxis.get_offset_text().set(size=8)
        ax.xaxis.set_tick_params(labelsize=8)

    return fig



def plot_error_vs_time(u_gtr: torch.Tensor,
                      u_prd: torch.Tensor,
                      t_values: np.array, 
                      t_indices:np.array, 
                      metadata: Metadata):
  """
    u_gtr: [num_of_samples, num_time_steps, num_of_points, num_channels]
    u_prd: [num_of_samples, num_time_steps, num_of_points, num_channels]
  """
  time_steps = u_prd.shape[1]
  errors = []
  for t in range(time_steps):
    u_gtr_t = u_gtr[:,t:t+1]
    u_prd_t = u_prd[:,t:t+1]

    error_t = compute_batch_errors(u_gtr_t, u_prd_t, metadata)
    error_t = compute_final_metric(error_t)
    errors.append(error_t)
  print(errors)
  print(t_values[t_indices])
  
  
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.plot(range(time_steps), errors, marker='o', label='Relative Error')
  ax.set_xlabel('Time Step')
  ax.set_ylabel('Relative Error')
  ax.set_title('Error vs Time Step')
  ax.legend()
  ax.grid(True)
  fig.savefig("test.jpg")
  return fig

def visualize_encoder_output(
    model: torch.nn.Module,
    rigraph,
    pndata: torch.Tensor,
    vis_compt: str = "encoder",
    channel_in: int = 0,
    domain: Tuple[List[float], List[float]] = ([0, 0], [1, 1])
) -> plt.Figure:
    """
    Visualizes the encoder output in a 4-column layout, matching the
    plotting style of 'plot_estimates' except for the actual content:

      1) Input (a single channel),
      2) First principal component (PC1),
      3) Second principal component (PC2),
      4) Scatter of PC1 vs. PC2 (no colorbar).

    Each of the first three columns has a horizontal colorbar with
    symmetric color limits (like in plot_estimates). The overall figure
    size, hatch fill, color scheme, and dynamic scatter sizing also
    replicate the plot_estimates aesthetic.

    Parameters
    ----------
    model: torch.nn.Module
        The whole (or similar) neural operator.
    rigraph : RegionInteractionGraph
        Graph object that has physical_to_regional.dst_ndata['pos'] as 2D coordinates.
    pndata : torch.Tensor
        Input data of shape [batch_size, n_nodes, in_channels].
        Only the first batch (index=0) is visualized.
    vis_compt: str, optional
        Which part of output you want to visualize.
    channel_in : int, optional
        Which input channel to visualize in the first column. Defaults to 0.
    domain : tuple of list, optional
        The displayed region as ([x_min, y_min], [x_max, y_max]).
        A hatched rectangle is drawn in each subplot to fill this domain,
        mirroring the style from plot_estimates.

    Returns
    -------
    fig : plt.Figure
        A Matplotlib Figure with 2 row-grids:
          - row 0 : four scatter subplots (Input, PC1, PC2, PC1 vs. PC2)
          - row 1 : four horizontal colorbars (the last one hidden)
        The rest of the styling (size, font, color scale) follows plot_estimates.

    Notes
    -----
    - We run encoder(...) under torch.no_grad() to obtain the latent representation
      of shape [B, n_nodes, latent_dim]. Then apply PCA(n_components=2) to the
      first batch's latent features for the second/third/fourth columns.
    - The first column's coloring uses pndata[0, :, channel_in], also plotted
      with a symmetrical color scale about zero, as in plot_estimates.
    - The scatter sizing is adapted from plot_estimates, scaling by the number
      of points to keep a roughly consistent density for different node counts.
    """

    # ~~~ 1) Forward pass + PCA ~~~
    model.eval()
    with torch.no_grad():
        if vis_compt == "encoder":
          encoded = model.encoder(rigraph, pndata)  # => [batch_size, n_nodes, latent_dim]
        if vis_compt == "processor":
          encoded = model.encoder(rigraph, pndata) 
          encoded = model.process(rigraph.regional_to_regional, encoded)
    # Only visualize batch=0
    encoded_0 = encoded[0]  # [n_nodes, latent_dim]
    # 2D node positions
    pos = rigraph.physical_to_regional.dst_ndata['pos'][:, :2].cpu().numpy()  # shape [n_nodes, 2]
    phy_pos = rigraph.physical_to_regional.src_ndata['pos'].cpu().numpy()

    # Input channel data
    inp_data = pndata[0, :, channel_in].cpu().numpy()  # [n_nodes]

    # PCA => 2 components
    encoded_0_np = encoded_0.cpu().numpy()
    pca = PCA(n_components=2)
    enc_pca = pca.fit_transform(encoded_0_np)  # => [n_nodes, 2]
    pc1, pc2 = enc_pca[:, 0], enc_pca[:, 1]

    # ~~~ 2) Create figure & Subplots ~~~
    #   - replicate the size logic from plot_estimates
    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = 0.2
    _SCATTER_SETTINGS_INPUT = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS_PCA = SCATTER_SETTINGS.copy()
    # dynamic sizing, as in plot_estimates (see the example)
    n_points = phy_pos.shape[0]
    _SCATTER_SETTINGS_INPUT['s'] = _SCATTER_SETTINGS_INPUT['s'] * 0.4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS_INPUT['s'] = _SCATTER_SETTINGS_INPUT['s'] * 128 / (n_points ** 0.5)

    n_points = pos.shape[0]
    _SCATTER_SETTINGS_PCA['s'] = _SCATTER_SETTINGS_PCA['s'] * 0.8 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS_PCA['s'] = _SCATTER_SETTINGS_PCA['s'] * 128 / (n_points ** 0.5)

    figsize = (8.6, _HEIGHT_PER_ROW + _HEIGHT_MARGIN)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(
        nrows=1,
        ncols=1,
        wspace=0.0,
        hspace=0.0
    )
    subfig = fig.add_subfigure(g_fig[0, 0], frameon=False)

    # Sub-gridspec (2 rows x 4 columns): row0=4 plots, row1=4 colorbars
    g = subfig.add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[1, 0.05],
        wspace=0.20,
        hspace=0.05
    )
    ax_inp  = subfig.add_subplot(g[0, 0])
    ax_pc1  = subfig.add_subplot(g[0, 1])
    ax_pc2  = subfig.add_subplot(g[0, 2])
    ax_pcs  = subfig.add_subplot(g[0, 3])

    cb_inp  = subfig.add_subplot(g[1, 0])
    cb_pc1  = subfig.add_subplot(g[1, 1])
    cb_pc2  = subfig.add_subplot(g[1, 2])
    cb_pcs  = subfig.add_subplot(g[1, 3])  # hidden later

    # ~~~ 3) Shared domain fill & axis style ~~~
    for ax in [ax_inp, ax_pc1, ax_pc2, ]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(domain[0][0], domain[1][0])
        ax.set_ylim(domain[0][1], domain[1][1])
        ax.fill_between(
            x=[domain[0][0], domain[1][0]],
            y1=domain[0][1],
            y2=domain[1][1],
            **HATCH_SETTINGS
        )
    ax_pcs.set_xticks([])
    ax_pcs.set_yticks([])

    # ~~~ 4) Column 1: Input data ~~~
    inp_vmax = float(np.max(inp_data))
    inp_vmin = float(np.min(inp_data))
    abs_vmax_inp = max(abs(inp_vmax), abs(inp_vmin))

    sc = ax_inp.scatter(
        phy_pos[:, 0], phy_pos[:, 1],
        c=inp_data,
        cmap=CMAP_BWR,
        vmin=-abs_vmax_inp,
        vmax=+abs_vmax_inp,
        **_SCATTER_SETTINGS_INPUT
    )
    cb = plt.colorbar(sc, cax=cb_inp, orientation='horizontal')
    cb.formatter.set_powerlimits((-0, 0))
    ax_inp.set_title("Input")

    # ~~~ 5) Column 2: PC1 ~~~
    pc1_vmax = float(pc1.max())
    pc1_vmin = float(pc1.min())
    abs_vmax_pc1 = max(abs(pc1_vmax), abs(pc1_vmin))

    sc = ax_pc1.scatter(
        pos[:, 0], pos[:, 1],
        c=pc1,
        cmap=CMAP_BWR,
        vmin=-abs_vmax_pc1,
        vmax=+abs_vmax_pc1,
        **_SCATTER_SETTINGS_PCA
    )
    cb = plt.colorbar(sc, cax=cb_pc1, orientation='horizontal')
    cb.formatter.set_powerlimits((-0, 0))
    ax_pc1.set_title("PCA1")

    # ~~~ 6) Column 3: PC2 ~~~
    pc2_vmax = float(pc2.max())
    pc2_vmin = float(pc2.min())
    abs_vmax_pc2 = max(abs(pc2_vmax), abs(pc2_vmin))

    sc = ax_pc2.scatter(
        pos[:, 0], pos[:, 1],
        c=pc2,
        cmap=CMAP_BWR,
        vmin=-abs_vmax_pc2,
        vmax=+abs_vmax_pc2,
        **_SCATTER_SETTINGS_PCA
    )
    cb = plt.colorbar(sc, cax=cb_pc2, orientation='horizontal')
    cb.formatter.set_powerlimits((-0, 0))
    ax_pc2.set_title("PCA2")

    # ~~~ 7) Column 4: (PC1, PC2) scatter, no colorbar ~~~
    ax_pcs.scatter(pc1, pc2, c='b', **_SCATTER_SETTINGS_PCA)
    ax_pcs.set_title("PCA1 vs. PCA2")
    cb_pcs.set_visible(False)

    return fig

