import os
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm, gamma
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import matplotlib.pyplot as plt
from typing import List, Any, Tuple

import math
import numpy as np
import matplotlib.pyplot as plt

def plot_mvn_data(
    x,
    theta,
    feature_names=None,
    alpha=0.7,
    s=12,
    figsize_per_plot=3.0,
    bins=30,
    hist_density=False,
):
    """
    Create pairwise scatter plots for x and theta; if d == 1, make histograms.

    Parameters
    ----------
    x : array-like, shape (n, d) or (n, 1, d)
        Observations.
    theta : array-like, shape (n, d) or (n, 1, d)
        Latent parameters.
    feature_names : list of str, optional
        Names for each dimension [0..d-1]. If None, uses ["0","1",...].
    alpha : float, optional
        Point transparency for scatter.
    s : float, optional
        Marker size for scatter.
    figsize_per_plot : float, optional
        Scales the overall figure size; each subplot gets roughly this width/height in inches.
    bins : int, optional
        Number of bins for histograms when d == 1.
    hist_density : bool, optional
        If True, normalize histogram to a density.

    Returns
    -------
    fig_x, fig_theta : matplotlib.figure.Figure
        The created figures.
    """

    # Convert to numpy
    x_np = np.asarray(x)
    th_np = np.asarray(theta)

    # Squeeze a singleton middle dimension if present: (n, 1, d) -> (n, d)
    if x_np.ndim == 3 and x_np.shape[1] == 1:
        x_np = np.squeeze(x_np, axis=1)
    if th_np.ndim == 3 and th_np.shape[1] == 1:
        th_np = np.squeeze(th_np, axis=1)

    if x_np.ndim != 2 or th_np.ndim != 2:
        raise ValueError("x and theta must be 2D with shape (n_samples, d) after squeezing.")

    if x_np.shape != th_np.shape:
        raise ValueError(f"x and theta must have the same shape, got {x_np.shape} vs {th_np.shape}.")

    n, d = x_np.shape

    # d == 1 -> histograms
    if d == 1:
        # x histogram
        fig_x, axx = plt.subplots(1, 1, figsize=(figsize_per_plot * 2.5, figsize_per_plot * 1.8))
        axx.hist(x_np[:, 0], bins=bins, density=hist_density, alpha=0.9)
        axx.set_xlabel("x[0]")
        axx.set_ylabel("Density" if hist_density else "Count")
        axx.set_title("Histogram of x")
        axx.grid(True, linestyle=":", linewidth=0.5)
        fig_x.tight_layout()

        # theta histogram
        fig_th, axt = plt.subplots(1, 1, figsize=(figsize_per_plot * 2.5, figsize_per_plot * 1.8))
        axt.hist(th_np[:, 0], bins=bins, density=hist_density, alpha=0.9)
        axt.set_xlabel("θ[0]")
        axt.set_ylabel("Density" if hist_density else "Count")
        axt.set_title("Histogram of θ")
        axt.grid(True, linestyle=":", linewidth=0.5)
        fig_th.tight_layout()

        return fig_x, fig_th

    # d >= 2 -> pairwise scatter plots
    # Dimension names
    if feature_names is None:
        feature_names = [str(i) for i in range(d)]
    if len(feature_names) != d:
        raise ValueError("feature_names length must match d.")

    # All unique pairs (i < j)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    n_plots = len(pairs)

    # Arrange subplots in a near-square grid
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    # --- Figure for x ---
    fig_x, axes_x = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot * n_cols, figsize_per_plot * n_rows),
        squeeze=False
    )
    for k, (i, j) in enumerate(pairs):
        r, c = divmod(k, n_cols)
        ax = axes_x[r, c]
        ax.scatter(x_np[:, i], x_np[:, j], alpha=alpha, s=s)
        ax.set_xlabel(f"x[{feature_names[i]}]")
        ax.set_ylabel(f"x[{feature_names[j]}]")
        ax.grid(True, linestyle=":", linewidth=0.5)
    # Hide any unused axes
    for k in range(n_plots, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes_x[r, c].set_visible(False)
    fig_x.suptitle("Pairwise scatter plots for x", y=0.995)
    fig_x.tight_layout()

    # --- Figure for theta ---
    fig_th, axes_th = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot * n_cols, figsize_per_plot * n_rows),
        squeeze=False
    )
    for k, (i, j) in enumerate(pairs):
        r, c = divmod(k, n_cols)
        ax = axes_th[r, c]
        ax.scatter(th_np[:, i], th_np[:, j], alpha=alpha, s=s)
        ax.set_xlabel(f"θ[{feature_names[i]}]")
        ax.set_ylabel(f"θ[{feature_names[j]}]")
        ax.grid(True, linestyle=":", linewidth=0.5)
    for k in range(n_plots, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes_th[r, c].set_visible(False)
    fig_th.suptitle("Pairwise scatter plots for θ", y=0.995)
    fig_th.tight_layout()

    return fig_x, fig_th
