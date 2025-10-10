import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
from typing import List

from data import mvn_posterior

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

def plot_multi_mvn_marginals(model,
                      params,
                      x_vals: jnp.ndarray,
                      prior_mean: jnp.ndarray,
                      prior_L: jnp.ndarray,
                      model_L: jnp.ndarray,
                      theta_range=(-10.0, 10.0),
                      num_points=300):
    """
    Plot learned vs true marginal CDFs for each dimension and a set of x's.

    Grid layout: rows=d (dimensions), cols=n (chosen x samples).

    Args
    ----
    model: vmapped MultiMDN with outputs (batch, d, K)
    params: model parameters
    x_vals: (n, d) subset of x to visualize (choose small n like 4–8)
    prior_mean: (d, 1)
    prior_L: (d, d)
    model_L: (d, d)
    theta_range: (min, max) for θ grid
    num_points: number of grid points for CDF curves

    Returns
    -------
    fig: matplotlib.figure.Figure
    axes: array of Axes with shape (d, n)
    """
    # Shapes and posterior
    n, d = x_vals.shape
    post_mean, post_var = mvn_posterior(x_vals, prior_mean, prior_L, model_L)  # (n,d), (d,d)
    post_std = jnp.sqrt(jnp.diag(post_var))             # (d,)

    # θ grid
    θ_grid = jnp.linspace(theta_range[0], theta_range[1], num_points)  # (P,)
    fig, axes = plt.subplots(d, n, sharey=True, figsize=(3.0*n, 2.6*d))
    if d==1:
        axes = axes[None,:]


    # Loop over chosen x's (columns)
    for col, x0 in enumerate(x_vals):
        x0_b = x0[None, ...]  # (1, d)
        logits, means, log_scales = model.apply(params, x0_b)  # each: (1, d, K)
        scales = jnp.exp(log_scales) 
        log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)  # (1, d, K)

        pi = jnp.exp(log_pi)[0]                                             # (d, K)
        means = means[0]    # (d, K)
        scales = scales[0]  # (d, K)

        # Loop over dimensions (rows)
        for row in range(d):
            ax = axes[row, col] if d > 1 else axes[0, col]

            # True marginal CDF N(μ_post[row], σ_post[row]^2)
            μp = post_mean[col, row]
            σp = post_std[row]
            # standardize and use standard-normal CDF
            true_cdf = norm.cdf((θ_grid - μp) / σp)  # (P,)

            # Learned mixture CDF
            μk = means[row]    # (K,)
            σk = scales[row]   # (K,)
            comp_cdfs = norm.cdf((θ_grid[:, None] - μk[None, :]) / σk[None, :])  # (P, K)
            learned_cdf = jnp.sum(pi[row] * comp_cdfs, axis=-1)                  # (P,)

            # Plot
            ax.plot(θ_grid, true_cdf, linestyle='--', linewidth=1.5, label='True CDF')
            ax.plot(θ_grid, learned_cdf, linewidth=1.5, label='Learned CDF')
            if col == 0:
                ax.set_ylabel(f"dim {row}")
            if row == d - 1:
                ax.set_xlabel("θ")
            if row == 0:
                ax.set_title(f"x={jnp.round(x0, 2)}")
            ax.grid(True, linestyle=":", linewidth=0.5)

            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    return fig, axes

def plot_mvn_marginals(model,
                      params: List,
                      x_vals: jnp.ndarray,
                      prior_mean: jnp.ndarray,
                      prior_L: jnp.ndarray,
                      model_L: jnp.ndarray,
                      theta_range=(-10.0, 10.0),
                      num_points=300,
                      path=None):
    """
    Plot learned vs true marginal CDFs for each dimension and a set of x's.

    Grid layout: rows=d (dimensions), cols=n (chosen x samples).

    Args
    ----
    model: vmapped MultiMDN with outputs (batch, d, K)
    params: model parameters
    x_vals: (n, d) subset of x to visualize (choose small n like 4–8)
    prior_mean: (d, 1)
    prior_L: (d, d)
    model_L: (d, d)
    theta_range: (min, max) for θ grid
    num_points: number of grid points for CDF curves

    Returns
    -------
    fig: matplotlib.figure.Figure
    axes: array of Axes with shape (d, n)
    """
    # Shapes and posterior
    n, d = x_vals.shape
    post_mean, post_var = mvn_posterior(x_vals, prior_mean, prior_L, model_L)  # (n,d), (d,d)
    post_std = jnp.sqrt(jnp.diag(post_var))             # (d,)

    # θ grid
    θ_grid = jnp.linspace(theta_range[0], theta_range[1], num_points)  # (P,)
    fig, axes = plt.subplots(d, n, sharey=True, figsize=(3.0*n, 2.6*d))
    if d==1:
        axes = axes[None,:]


    # Loop over chosen x's (columns)
    for col, x0 in enumerate(x_vals):
        x0_b = x0[None, ...]  # (1, d)
        
        # Loop over dimensions (rows)
        for row in range(d):
            ax = axes[row, col] if d > 1 else axes[0, col]

            logits, means, log_scales = model.apply(params[row], x0_b)  # each: (d, K)
            scales = jnp.exp(log_scales) 
            log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)  # (d, K)
            pi = jnp.exp(log_pi)                                           # (d, K)
          
            # True marginal CDF N(μ_post[row], σ_post[row]^2)
            μp = post_mean[col, row]
            σp = post_std[row]
            # standardize and use standard-normal CDF
            true_cdf = norm.cdf((θ_grid - μp) / σp)  # (P,)

            # Learned mixture CDF
            μk = means[row]    # (K,)
            σk = scales[row]   # (K,)
            comp_cdfs = norm.cdf((θ_grid[:, None] - μk[None, :]) / σk[None, :])  # (P, K)
            learned_cdf = jnp.sum(pi[row] * comp_cdfs, axis=-1)                  # (P,)

            # Plot
            ax.plot(θ_grid, true_cdf, linestyle='--', linewidth=1.5, label='True CDF')
            ax.plot(θ_grid, learned_cdf, linewidth=1.5, label='Learned CDF')
            if col == 0:
                ax.set_ylabel(f"dim {row}")
            if row == d - 1:
                ax.set_xlabel("θ")
            if row == 0:
                ax.set_title(f"x={jnp.round(x0, 2)}")
            ax.grid(True, linestyle=":", linewidth=0.5)

            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    plt.savefig(path + "fit.pdf")
    plt.close()


def plot_mdn_marginals(model,
                      params: List,
                      x_vals: jnp.ndarray,
                      theta_range=(-10.0, 10.0),
                      num_points=300,
                      path=None):
    """
    Plot learneds marginal CDFs for each dimension and a set of x's.

    Grid layout: rows=d (dimensions), cols=n (chosen x samples).

    Args
    ----
    model: vmapped MultiMDN with outputs (batch, d, K)
    params: model parameters
    x_vals: (n, d) subset of x to visualize (choose small n like 4–8)
    theta_range: (min, max) for θ grid
    num_points: number of grid points for CDF curves

    Returns
    -------
    fig: matplotlib.figure.Figure
    axes: array of Axes with shape (d, n)
    """
    # Shapes and posterior
    n, d = x_vals.shape

    # θ grid
    θ_grid = jnp.linspace(theta_range[0], theta_range[1], num_points)  # (P,)
    fig, axes = plt.subplots(d, n, sharey=True, figsize=(3.0*n, 2.6*d))
    if d==1:
        axes = axes[None,:]


    # Loop over chosen x's (columns)
    for col, x0 in enumerate(x_vals):
        x0_b = x0[None, ...]  # (1, d)
        
        # Loop over dimensions (rows)
        for row in range(d):
            ax = axes[row, col] if d > 1 else axes[0, col]

            logits, means, log_scales = model.apply(params[row], x0_b)  # each: (d, K)
            scales = jnp.exp(log_scales) 
            log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)  # (d, K)
            pi = jnp.exp(log_pi)                                           # (d, K)

            # Learned mixture CDF
            μk = means[row]    # (K,)
            σk = scales[row]   # (K,)
            comp_cdfs = norm.cdf((θ_grid[:, None] - μk[None, :]) / σk[None, :])  # (P, K)
            learned_cdf = jnp.sum(pi[row] * comp_cdfs, axis=-1)                  # (P,)

            # Plot
            ax.plot(θ_grid, learned_cdf, linewidth=1.5, label='Learned CDF')
            if col == 0:
                ax.set_ylabel(f"dim {row}")
            if row == d - 1:
                ax.set_xlabel("θ")
            if row == 0:
                ax.set_title(f"x={jnp.round(x0, 2)}")
            ax.grid(True, linestyle=":", linewidth=0.5)

            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    plt.savefig(path + "learned.pdf")
    plt.close()


def plot_losses(losses_list, path):

    n_plots = len(losses_list)

    # Arrange subplots in a near-square grid
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols)

    for idx, (row, col) in enumerate(product(range(n_rows), range(n_cols))):

            losses = losses_list[idx]
            ax = axes[row, col]
            n = len(losses)
            ns = [i for i in range(n)]

            ax.plot(ns, losses, color="red", linewidth=0.5)
            ax.set_title(f"θ dim {idx}")

            if row == n_rows-1:
                ax.set_xlabel("Training Step")
            if col == 0:    
                ax.set_ylabel("Training Loss")
    
    
    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "loss.pdf")
    plt.close()




def plot_loss(losses, path):

    fig, ax = plt.subplots(1,1)

    n = len(losses)
    ns = [i for i in range(n)]

    ax.plot(ns, losses, color="red", linewidth=0.5)
    ax.set_title("Training loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")

    plt.tight_layout()
    plt.savefig(path + "loss.pdf")
    plt.close()
    
