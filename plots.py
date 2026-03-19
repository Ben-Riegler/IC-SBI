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
from matplotlib.lines import Line2D
from DGPs.data import mvn_posterior
from tueplots.bundles import neurips2024

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern"],
# })

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False


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
        axt.set_xlabel(rf"$\theta$ [0]")
        axt.set_ylabel("Density" if hist_density else "Count")
        axt.set_title(rf"Histogram of $\theta$")
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
        ax.set_xlabel(rf"$\theta$ [{feature_names[i]}]")
        ax.set_ylabel(rf"$\theta$ [{feature_names[j]}]")
        ax.grid(True, linestyle=":", linewidth=0.5)
    for k in range(n_plots, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes_th[r, c].set_visible(False)
    fig_th.suptitle("Pairwise scatter plots for $\theta$", y=0.995)
    fig_th.tight_layout()

    return fig_x, fig_th

def plot_multi_mvn_marginals(model,
                      params,
                      x_vals: jnp.ndarray,
                      prior_mean: jnp.ndarray,
                      prior_L: jnp.ndarray,
                      model_L: jnp.ndarray,
                      theta_range=(-10.0, 10.0),
                      num_points=300,
                      save_path=None):
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
                ax.set_xlabel(rf"$\theta$")
            if row == 0:
                ax.set_title(f"x={jnp.round(x0, 2)}")
            ax.grid(True, linestyle=":", linewidth=0.5)

            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()

    name = "fit.pdf"

    if save_path is None:
        fig.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + name)
        plt.close(fig)
    

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
    model: MDN (batch, d, K)
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
    fig, axes = plt.subplots(d, n, sharey=True, figsize=(3.0*n, 2.6*d), layout="tight")
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
                ax.set_xlabel(rf"$\theta$")
            if row == 0:
                ax.set_title(f"x={jnp.round(x0, 2)}")
            ax.grid(True, linestyle=":", linewidth=0.5)

            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)

    # fig.tight_layout()
    plt.savefig(path + "fit.pdf")
    plt.close(fig)

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
    fig, axes = plt.subplots(d, n, sharey=True, figsize=(3.0*n, 2.6*d), layout="tight")
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
                ax.set_xlabel(rf"$\theta$")
            if row == 0:
                ax.set_title(f"x={jnp.round(x0, 2)}")
            ax.grid(True, linestyle=":", linewidth=0.5)

            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)

    # fig.tight_layout()
    if path is not None:
        plt.savefig(path + "learned.pdf")
        plt.close(fig)
    else:
        plt.show()

def plot_marginals_and_mdn(model,
                            params: List,
                            x_vals: jnp.ndarray,
                            sample:jnp.ndarray,
                            num_points=300,
                            title=rf"$p(y|x)$",
                            x_lab=rf"$\theta$",
                            path=None):
    """
    Plot learneds marginal CDFs for each dimension and a set of x's.

    Grid layout: rows=d (dimensions), cols=n (chosen x samples).

    Args
    ----
    model: vmapped MultiMDN with outputs (batch, d, K)
    params: model parameters
    x_vals: (n, x_dim)
    theta_range: (min, max) for θ grid
    num_points: number of grid points for CDF curves
    sample: (n, N, d)

    Returns
    -------
    fig: matplotlib.figure.Figure
    axes: array of Axes with shape (d, n)
    """
    # Shapes and posterior
    n, d = x_vals.shape


    mins = jnp.min(sample, axis=1) # (n, d)
    maxes = jnp.max(sample, axis=1) # (n, d)
    ranges = jnp.stack([mins, maxes], axis=-1) # (n, d, 2)

    
    grid = jnp.linspace(ranges[..., 0], ranges[..., 1], num_points, axis=-1)  # (n, d, num_points)
    fig, axes = plt.subplots(d, n, figsize=(3.0*n, 2.6*d), layout="tight")
    fig.suptitle(title)
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
            
            comp = norm.pdf((grid[col, row][:, None] - μk[None, :]) / σk[None, :]) /  σk[None, :] # (num_points, K)
            learned = jnp.sum(pi[row] * comp, axis=-1)                  # (num_points,))

            # Plot
            ax.plot(grid[col, row], learned, linewidth=1.5, label="learned density")
            ax.hist(sample[col, :,row], bins = 50, label="sample", density=True)
            if col == 0:
                ax.set_ylabel(f"dim {row+1}")
            if row == d - 1:
                ax.set_xlabel(x_lab)
            if row == 0:
                ax.set_title(f"x={jnp.round(x0, 2)}")
            ax.grid(True, linestyle=":", linewidth=0.5)

            if row == 0 and col == 0:
                ax.legend()

    # fig.tight_layout()
    if path is not None:
        plt.savefig(path + "samp_learned.pdf")
        plt.close(fig)
    else:
        plt.show()


def plot_losses(losses_list, path, val_losses_list = None):

    n_plots = len(losses_list)

    # Arrange subplots in a near-square grid
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, layout="tight")

    for idx, (row, col) in enumerate(product(range(n_rows), range(n_cols))):

            if idx < n_plots:
                losses = losses_list[idx]
                val_losses = val_losses_list[idx]
                ax = axes[row, col]
                n = len(losses)
                ns = [i for i in range(n)]

                label_train = "train loss" if idx == 0 else None
                label_val   = "val loss"   if idx == 0 else None

                ax.plot(ns, losses, color="red", linewidth=0.5, label=label_train)
                ax.set_title(rf"$\theta$ dim {idx}")

                if val_losses is not None:
                    ax.plot(ns, val_losses, color="blue", linewidth=0.5, label=label_val)



                if row == n_rows-1:
                    ax.set_xlabel("Training Step")
                if col == 0:    
                    ax.set_ylabel( "Loss")

                ax.set_ylim(top = min(5, max(losses)))
    
    fig.legend()
    
    # plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "loss.pdf")
    plt.close(fig)

def plot_loss(losses, path, val_losses = None):

    fig, ax = plt.subplots(1,1, layout="tight")

    n = len(losses)
    ns = [i for i in range(n)]

    ax.plot(ns, losses, color="red", linewidth=0.5, label="training loss")

    if val_losses is not None:
        ax.plot(ns, val_losses, color="blue", linewidth=0.5, label="validation loss")
        ax.set_title("Training and Validation")
    else:
        ax.set_title("Training Loss")

    
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")

    ax.legend()
    # plt.tight_layout()
    plt.savefig(path + "loss.pdf")
    plt.close(fig)

def plot_post_pairs(
    theta,
    theta_true=None,
    post_mean=None,
    post_cov=None,
    save_dir=None,
    legend=True,
    title = "",
    ax_lab=rf"\theta",
    x_vals=None,
    file_prefix="pairplot",
    max_scatter=None,
    grid_size=200,
    chi2_probs=(0.50, 0.75, 0.90),
    k_sigma=3.0,
    dpi=150,
    seed=0,
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    theta = np.asarray(theta)
    if post_mean is not None:
        post_mean = np.asarray(post_mean)
        if post_mean.ndim == 3 and post_mean.shape[-1] == 1:
            post_mean = post_mean[..., 0]
        post_cov = np.asarray(post_cov)

    n_x, n_samps, d = theta.shape
    if post_mean is not None:
        assert post_mean.shape == (n_x, d)
        assert post_cov.shape == (n_x, d, d)
    if d < 2:
        raise ValueError("d must be >= 2 to form pairwise plots.")

    chi2_q = {
        0.50: 1.38629436112,
        0.75: 2.77258872224,
        0.90: 4.60517018599,
        0.95: 5.99146454711,
        0.99: 9.21034037198,
    }
    chi2_vals = [chi2_q[p] for p in chi2_probs]

    for i in range(n_x):
        # widen width to make room for the legend on the right
        fig_w = 3.0 * (d - 1) + 1.25
        fig_h = 3.0 * (d - 1) + 0.25
        fig, axes = plt.subplots(
            d - 1, d - 1,
            figsize=(fig_w, fig_h),
            layout="tight", 
        )
        if d - 1 == 1:
            axes = np.array([[axes]])

        if post_mean is not None:
            mu_full = post_mean[i]
            Sigma_full = post_cov[i]

        sel = rng.choice(n_samps, size=min(n_samps, max_scatter), replace=False) if max_scatter is not None else jnp.arange(n_samps)

        # ensure legend items appear only once per figure
        show_approx_legend = True
        show_true_legend = True
        cs_for_legend = None  # first contour set (for legend styles)

        for r in range(d - 1):
            for c in range(d - 1):
                ax = axes[r, c]
                if r < c:
                    ax.axis("off")
                    continue

                a, b = c, r + 1

                xs = theta[i, sel, a]
                ys = theta[i, sel, b]
                ax.scatter(
                    xs, ys, s=4, alpha=0.35, linewidths=0,
                    label=("approx samples" if show_approx_legend else "_nolegend_"),
                )
                show_approx_legend = False

                if theta_true is not None:
                    xs_tr = theta_true[i, sel, a]
                    ys_tr = theta_true[i, sel, b]
                    ax.scatter(
                        xs_tr, ys_tr, s=4, alpha=0.2, linewidths=0, color="red",
                        label=("true samples" if show_true_legend else "_nolegend_"),
                    )
                    show_true_legend = False

                if post_mean is not None:
                    mu = mu_full[[a, b]]
                    Sigma = Sigma_full[np.ix_([a, b], [a, b])]
                    det = np.linalg.det(Sigma)
                    if not np.isfinite(det) or det <= 0:
                        jitter = 1e-6 * (np.trace(Sigma) / 2.0 if np.isfinite(np.trace(Sigma)) else 1.0)
                        Sigma = Sigma + jitter * np.eye(2)

                    inv = np.linalg.inv(Sigma)
                    std = np.sqrt(np.diag(Sigma))

                    x_min = np.minimum(xs.min(), mu[0] - k_sigma * std[0])
                    x_max = np.maximum(xs.max(), mu[0] + k_sigma * std[0])
                    y_min = np.minimum(ys.min(), mu[1] - k_sigma * std[1])
                    y_max = np.maximum(ys.max(), mu[1] + k_sigma * std[1])

                    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                        pad = 1e-3 if np.isfinite(mu[0]) else 1.0
                        x_min, x_max = mu[0] - pad, mu[0] + pad
                    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                        pad = 1e-3 if np.isfinite(mu[1]) else 1.0
                        y_min, y_max = mu[1] - pad, mu[1] + pad

                    xx = np.linspace(x_min, x_max, grid_size)
                    yy = np.linspace(y_min, y_max, grid_size)
                    XX, YY = np.meshgrid(xx, yy)
                    DX = XX - mu[0]
                    DY = YY - mu[1]

                    R = DX * (inv[0, 0] * DX + inv[0, 1] * DY) + DY * (inv[1, 0] * DX + inv[1, 1] * DY)
                    cs = ax.contour(XX, YY, R, levels=chi2_vals, linewidths=1.0)

                    if cs_for_legend is None:
                        cs_for_legend = cs

                if c == 0:
                    ax.set_ylabel(rf"${ax_lab}_{{{b+1}}}$")
                if r == d - 2:
                    ax.set_xlabel(rf"${ax_lab}_{{{a+1}}}$")

        # title (reserve space above with tight_layout rect)
        if x_vals is not None:
            x_str = np.array2string(np.asarray(x_vals[i]), precision=2, separator=", ")
            title_ =  title + f"x = {x_str}"
        fig.suptitle(title_, fontsize=10)

        # --- Build a clean figure legend (per figure) and park it to the right
        handles = [Line2D([], [], marker="o", linestyle="None", markersize=4, alpha=0.35, label="approx samples")]
        if theta_true is not None:
            handles.append(Line2D([], [], marker="o", linestyle="None", markersize=4, alpha=0.2, color="red", label="true samples"))

        if cs_for_legend is not None:
            contour_handles, _ = cs_for_legend.legend_elements()
            for h, p in zip(contour_handles[:len(chi2_probs)], chi2_probs):
                h.set_label(f"{int(round(p * 100))}% level")
                handles.append(h)

        if legend:
            # place legend outside the plotting grid, so it never overlaps the title
            fig.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                title="Legend",
            )

        # fig.tight_layout() 

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"{file_prefix}_{i}.png")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            fig.show()


# plt.rcParams.update(neurips2024(nrows=2, ncols=3))
# plt.rcParams.update({'figure.figsize': (6.75, 1.6)})


def plot_post_pairs_triptych(
    theta_list,                   # list of 3 (n_x, n_samps, d) arrays
    theta_true=None,              # single or list[3] like theta
    post_mean=None,               # single or list[3]: (n_x, d) or None
    post_cov=None,                # single or list[3]: (n_x, d, d) or None
    save_dir=None,
    legend=True,
    title="",
    ax_lab=rf"\theta",            # str or list/tuple of 3 strings
    x_vals=None,                  # single or list[3]; each either (n_x, *) or None
    file_prefix="pairplot",
    max_scatter=None,
    grid_size=200,
    chi2_probs=(0.50, 0.75, 0.90),
    k_sigma=3.0,
    dpi=150,
    pad=0, 
    y_title=1,
    seed=0,
    titles=["", "", ""],          # per-panel titles
    # spacing controls
    inner_wspace=0.30,            # inside each (d-1)x(d-1) grid
    inner_hspace=0.18,
    panel_wspace=0.25,            # NEW: spacing between the three panels (was fixed)
    legend_bottom_margin=0.22,    # CHANGED: a bit more space reserved for legend
):
    def _as_list(x, name):
        if isinstance(x, (list, tuple)):
            assert len(x) == 3, f"{name} must be length-3 if a list/tuple."
            return list(x)
        return [x, x, x]

    def _get_axlab(k):
        if isinstance(ax_lab, (list, tuple)):
            return ax_lab[k]
        return ax_lab

    def _panel_caption(base, x_i):
        if x_i is None or base is None or base == "":
            return base or ""
        x_arr = np.asarray(x_i)
        return base + f"x = {np.array2string(x_arr, precision=2, separator=', ')}"

    thetas = list(theta_list)
    assert len(thetas) == 3, "theta_list must have length 3."

    theta_true_list = _as_list(theta_true, "theta_true")
    post_mean_list  = _as_list(post_mean, "post_mean")
    post_cov_list   = _as_list(post_cov, "post_cov")
    x_vals_list     = _as_list(x_vals, "x_vals")

    n_xs, ds = [], []
    for k in range(3):
        th = np.asarray(thetas[k])
        if th.ndim != 3:
            raise ValueError(f"theta_list[{k}] must have shape (n_x, n_samps, d)")
        n_x, n_samps, d = th.shape
        n_xs.append(n_x); ds.append(d)

        pm = post_mean_list[k]
        pc = post_cov_list[k]
        if pm is not None:
            pm = np.asarray(pm)
            if pm.ndim == 3 and pm.shape[-1] == 1:
                pm = pm[..., 0]
            if pm.shape != (n_x, d):
                raise ValueError(f"post_mean[{k}] must have shape (n_x, d); got {pm.shape}")
            post_mean_list[k] = pm
            if pc is None:
                raise ValueError(f"post_cov[{k}] must be provided when post_mean[{k}] is provided.")
            pc = np.asarray(pc)
            if pc.shape != (n_x, d, d):
                raise ValueError(f"post_cov[{k}] must have shape (n_x, d, d); got {pc.shape}")
            post_cov_list[k] = pc
        else:
            post_mean_list[k] = None

    if not (n_xs[0] == n_xs[1] == n_xs[2]):
        raise ValueError(f"All thetas must share the same n_x; got {n_xs}.")
    if not (ds[0] == ds[1] == ds[2]):
        raise ValueError(f"All thetas must share the same dimensionality d; got {ds}.")
    n_x = n_xs[0]
    d = ds[0]
    if d < 2:
        raise ValueError("d must be >= 2 to form pairwise plots.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    chi2_q = {
        0.50: 1.38629436112,
        0.75: 2.77258872224,
        0.90: 4.60517018599,
        0.95: 5.99146454711,
        0.99: 9.21034037198,
    }
    chi2_vals = [chi2_q[p] for p in chi2_probs]

    for i in range(n_x):
        # Keep your overall sizing logic; square-ness is enforced by set_box_aspect(1)
        single_w = 3.0 * (d - 1)
        single_h = 3.0 * (d - 1) + 0.25

        fig_w = 3 * single_w
        fig_h = single_h

        # fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
        fig = plt.figure(layout="tight")

        gs = fig.add_gridspec(
            nrows=1, ncols=3,
            width_ratios=[single_w, single_w, single_w],
            wspace=panel_wspace,           # CHANGED
        )

        show_approx_legend = True
        show_true_legend = True
        cs_for_legend = None

        for k in range(3):
            th_k = np.asarray(thetas[k])
            _, n_samps_k, _ = th_k.shape
            pm_k = post_mean_list[k]
            pc_k = post_cov_list[k]
            tt_k = theta_true_list[k]
            xv_k = x_vals_list[k]

            host_ax = fig.add_subplot(gs[0, k])
            host_ax.set_title(titles[k],pad=pad, loc="center", y=y_title)
            host_ax.axis("off")

            sub_gs = host_ax.get_subplotspec().subgridspec(
                d - 1, d - 1, wspace=inner_wspace, hspace=inner_hspace
            )

            axes = np.empty((d - 1, d - 1), dtype=object)
            for r in range(d - 1):
                for c in range(d - 1):
                    axes[r, c] = fig.add_subplot(sub_gs[r, c])

            if xv_k is not None:
                try:
                    xv_i = xv_k[i]
                except Exception:
                    xv_i = xv_k
                axes[0, 0].set_title(_panel_caption(title, xv_i), loc="left", pad=pad)

            sel = rng.choice(n_samps_k, size=min(n_samps_k, max_scatter), replace=False) if max_scatter is not None else np.arange(n_samps_k)

            if pm_k is not None:
                mu_full = pm_k[i]
                Sigma_full = pc_k[i]

            for r in range(d - 1):
                for c in range(d - 1):
                    ax = axes[r, c]
                    if r < c:
                        ax.axis("off")
                        continue

                    # --- make each small axes square in display space ---
                    ax.set_box_aspect(1)      # NEW: square boxes regardless of data scales

                    a, b = c, r + 1

                    xs = th_k[i, sel, a]
                    ys = th_k[i, sel, b]
                    ax.scatter(
                        xs, ys, s=1, alpha=0.35, linewidths=0,
                        label=("approx samples" if show_approx_legend else "_nolegend_"),
                    )
                    show_approx_legend = False

                    if tt_k is not None:

                        from scipy.spatial import ConvexHull

                        # Suppose these come from your JAX arrays
                        # xs_tr = np.asarray(tt_k[i, sel, a])
                        # ys_tr = np.asarray(tt_k[i, sel, b])

                        # # Stack into (N, 2) array of points
                        # points = np.column_stack((xs_tr, ys_tr))

                        # # Compute convex hull
                        # hull = ConvexHull(points)

                        # # Plot the scatter and the convex hull
                        # ax.scatter(xs_tr, ys_tr, s=1, alpha=0.2, linewidths=0, color="red",
                        #            label=("true samples" if show_true_legend else "_nolegend_")
                        #            )

                        # # Add hull outline
                        # for simplex in hull.simplices:
                        #     ax.plot(points[simplex, 0], points[simplex, 1], "r-", lw=1)

                        xs_tr = tt_k[i, sel, a]
                        ys_tr = tt_k[i, sel, b]
                        ax.scatter(
                            xs_tr, ys_tr, s=1, alpha=0.2, linewidths=0, color="red",
                            label=("True Posterior Samples" if show_true_legend else "_nolegend_"),
                        )
                        import matplotlib.ticker as mticker

                        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
                        show_true_legend = False

                    if pm_k is not None:
                        mu = mu_full[[a, b]]
                        Sigma = Sigma_full[np.ix_([a, b], [a, b])]
                        det = np.linalg.det(Sigma)
                        if not np.isfinite(det) or det <= 0:
                            jitter = 1e-6 * (np.trace(Sigma) / 2.0 if np.isfinite(np.trace(Sigma)) else 1.0)
                            Sigma = Sigma + jitter * np.eye(2)

                        inv = np.linalg.inv(Sigma)
                        std = np.sqrt(np.diag(Sigma))

                        x_min = np.minimum(xs.min(), mu[0] - k_sigma * std[0])
                        x_max = np.maximum(xs.max(), mu[0] + k_sigma * std[0])
                        y_min = np.minimum(ys.min(), mu[1] - k_sigma * std[1])
                        y_max = np.maximum(ys.max(), mu[1] + k_sigma * std[1])

                        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                            pad = 1e-3 if np.isfinite(mu[0]) else 1.0
                            x_min, x_max = mu[0] - pad, mu[0] + pad
                        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                            pad = 1e-3 if np.isfinite(mu[1]) else 1.0
                            y_min, y_max = mu[1] - pad, mu[1] + pad

                        xx = np.linspace(x_min, x_max, grid_size)
                        yy = np.linspace(y_min, y_max, grid_size)
                        XX, YY = np.meshgrid(xx, yy)
                        DX = XX - mu[0]
                        DY = YY - mu[1]
                        R = DX * (inv[0, 0] * DX + inv[0, 1] * DY) + DY * (inv[1, 0] * DX + inv[1, 1] * DY)
                        cs = ax.contour(XX, YY, R, levels=chi2_vals, linewidths=1.0)

                        if cs_for_legend is None:
                            cs_for_legend = cs

                    lab = _get_axlab(k)
                    if c == 0:
                        ax.set_ylabel(rf"${lab}_{{{b+1}}}$")
                        ax.tick_params(axis="y", which="both", left=True, labelleft=True)
                    else:
                        ax.set_ylabel("")  # no label
                        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                    if r == d - 2:
                        ax.set_xlabel(rf"${lab}_{{{a+1}}}$")
                        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
                    else:
                        ax.set_xlabel("")  # no label
                        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Build legend handles (unchanged)
        handles = [Line2D([], [], marker="o", linestyle="None", markersize=4, alpha=0.35, label="Samples")]
        if any(t is not None for t in theta_true_list):
            handles.append(Line2D([], [], marker="o", linestyle="None", markersize=4, alpha=0.2, color="red", label="True Posterior Samples"))

        if cs_for_legend is not None:
            contour_handles, _ = cs_for_legend.legend_elements()
            for h, p in zip(contour_handles[:len(chi2_probs)], chi2_probs):
                h.set_label(f"{int(round(p * 100))}% level")
                handles.append(h)

        if legend:
            # Reserve a bigger strip at the bottom and place legend centered within it.
            fig.tight_layout() # rect=[0.02, legend_bottom_margin, 0.98, 0.98]
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, legend_bottom_margin/2),  # NEW: tie to reserved margin
                ncol=min(len(handles), 6),
                frameon=False,
                borderaxespad=0.0,
                handletextpad=0.6,
                columnspacing=1.2,
                labelspacing=0.6,
            )
        # else:
        #     fig.tight_layout()

        if title:
            fig.suptitle(title, y = y_title)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{file_prefix}_{i}.pdf")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            fig.show()

def plot_marginal_hists(u, # (batch, N, d)
                        x, # (batch, d)
                        save_path=None,
                        title=rf"Histograms of $U_j = F_j(Y_j \mid x)$",
                        name="marg_hists.pdf",
                        bins=50
                        ):

    locs, N, d = u.shape
    fig, axes = plt.subplots(d, locs, figsize=(3*locs, 4*d), constrained_layout=True)

    for row in range(d):
        for col in range(locs):
            ax = axes[row, col]

            ax.hist(u[col, :, row], bins=bins, density=True)

            if col == 0:
                ax.set_ylabel(f"dim {row+1}")
            
            if row == 0:
                ax.set_title(f"x = {jnp.round(x[col], 2)}")
    
    fig.suptitle(title)

    if save_path is None:
        fig.show()
    else:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + name)
        plt.close(fig)
