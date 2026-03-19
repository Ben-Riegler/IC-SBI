"""
abc_reference.py

Approximate Bayesian Computation (ABC) rejection sampler for the mixed DGP:

    t1 ~ N(0, 1)
    t2 ~ Gamma(1, 1)
    t3 ~ Beta(2, 3)
    c  ~ Bin(20, t3)
    z  ~ N(t1, sqrt(t2))
    x  = c + z

Since x is a scalar, we use a simple rejection ABC with |x_sim - x_obs| < eps.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Vectorised simulator (draws a big batch of prior samples + simulated x)
# ---------------------------------------------------------------------------

def simulate_batch(key, n):
    """
    Draw n samples from the joint (theta, x).
    Returns
        theta: (n, 3)   [t1, t2, t3]
        x_sim: (n,)
    """
    k1, k2, k3, k4, k5 = random.split(key, 5)

    t1 = random.normal(k1, (n,))
    t2 = random.gamma(k2, 1.0, (n,))
    t3 = random.beta(k3, 2.0, 3.0, shape=(n,))

    c = random.binomial(k4, n=20, p=t3).astype(t1.dtype)
    z = t1 + jnp.sqrt(t2) * random.normal(k5, (n,))

    x_sim = c + z
    theta = jnp.stack([t1, t2, t3], axis=-1)
    return theta, x_sim


# ---------------------------------------------------------------------------
# ABC rejection sampler
# ---------------------------------------------------------------------------

def abc_rejection(key, x_obs, n_accept, eps,
                  batch_size=1_000_000, max_batches=200):
    """
    Rejection ABC for a single observed x_obs (scalar).

    Parameters
    ----------
    key        : PRNG key
    x_obs      : float, the observed data point
    n_accept   : int, desired number of accepted posterior samples
    eps        : float, acceptance threshold  |x_sim - x_obs| < eps
    batch_size : int, how many prior samples to simulate per batch
    max_batches: int, safety limit

    Returns
    -------
    theta_accepted : (n_accepted, 3)   — may be >= n_accept
    n_total        : int, total simulations used
    accept_rate    : float
    """
    accepted = []
    n_total = 0

    for _ in range(max_batches):
        key, sk = random.split(key)
        theta_batch, x_batch = simulate_batch(sk, batch_size)
        n_total += batch_size

        # rejection criterion
        dist = jnp.abs(x_batch - x_obs)
        mask = dist < eps

        theta_keep = theta_batch[mask]
        if theta_keep.shape[0] > 0:
            accepted.append(np.asarray(theta_keep))  # move to CPU

        n_so_far = sum(a.shape[0] for a in accepted)
        if n_so_far >= n_accept:
            break

    theta_accepted = np.concatenate(accepted, axis=0)[:n_accept]
    accept_rate = n_accept / n_total
    return jnp.array(theta_accepted), n_total, accept_rate


def abc_rejection_multi(key, x_obs_batch, n_accept, eps,
                        batch_size=1_000_000, max_batches=200):
    """
    Run ABC rejection for multiple observed x values.

    Parameters
    ----------
    key          : PRNG key
    x_obs_batch  : (n_obs, 1) or (n_obs,)  — multiple observations
    n_accept     : int, desired accepted samples per observation
    eps          : float, acceptance threshold
    batch_size   : int
    max_batches  : int

    Returns
    -------
    theta_all : (n_obs, n_accept, 3)
    """
    x_obs_batch = jnp.atleast_1d(x_obs_batch.squeeze())
    n_obs = x_obs_batch.shape[0]
    results = []

    for i in range(n_obs):
        key, sk = random.split(key)
        x_o = float(x_obs_batch[i])
        theta_acc, n_tot, rate = abc_rejection(
            sk, x_o, n_accept, eps, batch_size, max_batches
        )
        print(f"  x_obs[{i}] = {x_o:+.3f}  |  "
              f"accepted {theta_acc.shape[0]}/{n_tot}  "
              f"(rate={rate:.2e})")
        results.append(theta_acc)

    return jnp.stack(results, axis=0)  # (n_obs, n_accept, 3)


# ---------------------------------------------------------------------------
# Adaptive epsilon selection
# ---------------------------------------------------------------------------

def calibrate_eps(key, quantile=0.001, pilot_n=1_000_000):
    """
    Run a pilot simulation and pick eps as a quantile of the
    prior-predictive |x - median(x)| distribution.
    Gives a rough sense of scale.
    """
    _, x_pilot = simulate_batch(key, pilot_n)
    x_median = jnp.median(x_pilot)
    dists = jnp.abs(x_pilot - x_median)
    eps = float(jnp.quantile(dists, quantile))
    print(f"Pilot calibration: median(x)={x_median:.3f}, "
          f"eps(q={quantile})={eps:.4f}")
    return eps


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_abc_posterior(theta_abc, x_obs_val, save_path=None):
    """
    Plot marginal histograms and pair plots of ABC posterior samples.

    theta_abc : (n_obs, n_accept, 3)
    x_obs_val : (n_obs,) or (n_obs, 1)
    """
    x_obs_val = jnp.atleast_1d(x_obs_val.squeeze())
    n_obs = theta_abc.shape[0]
    names = [r"$\theta_1$ (Normal)", r"$\theta_2$ (Gamma)", r"$\theta_3$ (Beta)"]

    # --- Marginal histograms ---
    fig, axes = plt.subplots(n_obs, 3, figsize=(12, 3 * n_obs))
    if n_obs == 1:
        axes = axes[None, :]
    for i in range(n_obs):
        for j in range(3):
            axes[i, j].hist(np.array(theta_abc[i, :, j]),
                            bins=50, density=True, alpha=0.7)
            axes[i, j].set_title(f"x={float(x_obs_val[i]):.2f}  {names[j]}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "abc_marginals.pdf")
    plt.show()
    plt.close()

    # --- Pair plots ---
    import itertools as it
    pairs = list(it.combinations(range(3), 2))
    for i in range(min(n_obs, 4)):  # plot at most 4 observations
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
        for ax, (a, b) in zip(axes, pairs):
            ax.scatter(np.array(theta_abc[i, :, a]),
                       np.array(theta_abc[i, :, b]),
                       s=1, alpha=0.3)
            ax.set_xlabel(names[a])
            ax.set_ylabel(names[b])
        fig.suptitle(f"ABC posterior  x_obs={float(x_obs_val[i]):.2f}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + f"abc_pairs_obs{i}.pdf")
        plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_obs", type=int, default=5,
                        help="Number of test x observations")
    parser.add_argument("--n_accept", type=int, default=2000,
                        help="Desired posterior samples per observation")
    parser.add_argument("--eps", type=float, default=0.01,
                        help="Acceptance threshold (auto-calibrated if None)")
    parser.add_argument("--eps_quantile", type=float, default=0.001,
                        help="Quantile for auto eps calibration")
    parser.add_argument("--batch_size", type=int, default=2_000_000)
    parser.add_argument("--save_path", type=str, default="abc_reference/")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    key = random.PRNGKey(args.seed)

    # generate some test observations from the DGP
    key, dk = random.split(key)
    from data import gen_mixed_prior_binomial_normal_data
    x_all, theta_all = gen_mixed_prior_binomial_normal_data(dk, args.n_obs, True)
    x_obs = x_all.squeeze()  # (n_obs,)
    theta_true = theta_all   # (n_obs, 3) — the true parameters that generated x

    print(f"\nTest observations: {x_obs}")
    print(f"True parameters:\n{theta_true}\n")

    # calibrate epsilon
    if args.eps is None:
        key, ck = random.split(key)
        eps = calibrate_eps(ck, quantile=args.eps_quantile)
    else:
        eps = args.eps
    print(f"Using eps = {eps:.6f}\n")

    # run ABC
    t0 = time.perf_counter()
    key, ak = random.split(key)
    theta_abc = abc_rejection_multi(
        ak, x_obs,
        n_accept=args.n_accept,
        eps=eps,
        batch_size=args.batch_size,
    )
    t1 = time.perf_counter()
    print(f"\nABC sampling took {t1 - t0:.1f}s")
    print(f"Result shape: {theta_abc.shape}")  # (n_obs, n_accept, 3)

    # save
    np.savez_compressed(
        os.path.join(args.save_path, "abc_samples.npz"),
        theta_abc=np.asarray(theta_abc),
        x_obs=np.asarray(x_obs),
        theta_true=np.asarray(theta_true),
        eps=eps,
    )

    # plot
    plot_abc_posterior(theta_abc, x_obs, save_path=args.save_path)

    # print summary statistics
    print("\n--- ABC posterior summaries ---")
    for i in range(x_obs.shape[0]):
        print(f"\nx_obs = {float(x_obs[i]):.3f}  "
              f"(true θ = {np.array(theta_true[i])})")
        for j, name in enumerate(["t1", "t2", "t3"]):
            samples = np.array(theta_abc[i, :, j])
            print(f"  {name}: mean={samples.mean():.4f}  "
                  f"std={samples.std():.4f}  "
                  f"95% CI=[{np.percentile(samples, 2.5):.4f}, "
                  f"{np.percentile(samples, 97.5):.4f}]")
