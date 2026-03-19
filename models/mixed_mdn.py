import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
from typing import List, Tuple, Any
import time

from utils import save_multiMDN

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Log-pdf helpers
# ---------------------------------------------------------------------------

def gaussian_log_prob(y, mu, sigma):
    """
    y:     (...,)
    mu:    (..., K)
    sigma: (..., K)
    returns: (..., K)
    """
    return (
        -0.5 * ((y[..., None] - mu) / sigma) ** 2
        - jnp.log(sigma)
        - 0.5 * jnp.log(2 * jnp.pi)
    )


def gamma_log_prob(y, alpha, beta):
    """
    Gamma(y; alpha, beta) with shape alpha, rate beta.
    y:     (...,)
    alpha: (..., K)
    beta:  (..., K)
    returns: (..., K)
    """
    y_k = y[..., None]  # (..., 1)
    y_k = jnp.clip(y_k, 1e-12)
    return (
        alpha * jnp.log(beta)
        - jax.lax.lgamma(alpha)
        + (alpha - 1.0) * jnp.log(y_k)
        - beta * y_k
    )


def beta_log_prob(y, a, b):
    """
    Beta(y; a, b).
    y: (...,)
    a: (..., K)
    b: (..., K)
    returns: (..., K)
    """
    y_k = y[..., None]  # (..., 1)
    y_k = jnp.clip(y_k, 1e-12, 1.0 - 1e-12)
    return (
        (a - 1.0) * jnp.log(y_k)
        + (b - 1.0) * jnp.log(1.0 - y_k)
        + jax.lax.lgamma(a + b)
        - jax.lax.lgamma(a)
        - jax.lax.lgamma(b)
    )


# ---------------------------------------------------------------------------
# CDF helpers  (used by PIT and inversion)
# ---------------------------------------------------------------------------

def gaussian_cdf(y, mu, sigma):
    """Component CDFs for Gaussian. returns (..., K)"""
    from jax.scipy.stats import norm
    return norm.cdf((y[..., None] - mu) / sigma)


def gamma_cdf(y, alpha, beta):
    """Component CDFs for Gamma(alpha, beta). returns (..., K)"""
    y_k = jnp.clip(y[..., None], 1e-30)
    y_k = y_k.astype(alpha.dtype)  # ensure same dtype
    return jax.lax.igamma(alpha, beta * y_k)


def beta_cdf(y, a, b):
    """Component CDFs for Beta(a, b). returns (..., K)"""
    y_k = jnp.clip(y[..., None], 1e-12, 1.0 - 1e-12)  # 1e-12 and 1.0 are float64!
    y_k = y_k.astype(a.dtype)
    return jax.lax.betainc(a, b, y_k)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MixedMarginalMDN(nn.Module):
    """
    Shared encoder → three independent mixture heads:
        dim 0  (t1):  mixture of Gaussians   → (logits, mu, log_sigma)
        dim 1  (t2):  mixture of Gammas       → (logits, log_alpha, log_beta)
        dim 2  (t3):  mixture of Betas        → (logits, log_a, log_b)
    """
    hidden_dims: List[int]
    K: int  # number of mixture components per head

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        x: (B, x_dim)
        Returns dict with keys 'gaussian', 'gamma', 'beta',
        each a tuple of (logits, param1, param2) all (B, K).
        """
        h = x
        for dim in self.hidden_dims[:-1]:
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
        h = nn.Dense(self.hidden_dims[-1])(h)
        h = nn.relu(h)

        # --- Gaussian head (t1) ---
        g_logits    = nn.Dense(self.K, name="g_logits")(h)      # (B, K)
        g_mu        = nn.Dense(self.K, name="g_mu")(h)           # (B, K)
        g_log_sigma = nn.Dense(self.K, name="g_log_sigma")(h)   # (B, K)

        # --- Gamma head (t2) ---
        ga_logits    = nn.Dense(self.K, name="ga_logits")(h)     # (B, K)
        ga_log_alpha = nn.Dense(self.K, name="ga_log_alpha")(h)  # (B, K)
        ga_log_beta  = nn.Dense(self.K, name="ga_log_beta")(h)   # (B, K)

        # --- Beta head (t3) ---
        b_logits = nn.Dense(self.K, name="b_logits")(h)          # (B, K)
        b_log_a  = nn.Dense(self.K, name="b_log_a")(h)           # (B, K)
        b_log_b  = nn.Dense(self.K, name="b_log_b")(h)           # (B, K)

        return {
            "gaussian": (g_logits, g_mu, g_log_sigma),
            "gamma":    (ga_logits, ga_log_alpha, ga_log_beta),
            "beta":     (b_logits, b_log_a, b_log_b),
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def mixed_mdn_log_prob(out_dict, theta):
    """
    out_dict: output of MixedMarginalMDN
    theta:    (B, 3)  — columns [t1, t2, t3]
    returns:  scalar  (mean negative log-likelihood)
    """
    t1, t2, t3 = theta[:, 0], theta[:, 1], theta[:, 2]

    # --- Gaussian (t1) ---
    g_logits, g_mu, g_log_sigma = out_dict["gaussian"]
    g_log_pi = g_logits - jax.nn.logsumexp(g_logits, axis=-1, keepdims=True)
    g_sigma = jnp.exp(g_log_sigma)
    g_comp = gaussian_log_prob(t1, g_mu, g_sigma)             # (B, K)
    ll_t1 = jax.nn.logsumexp(g_log_pi + g_comp, axis=-1)     # (B,)

    # --- Gamma (t2) ---
    ga_logits, ga_log_alpha, ga_log_beta = out_dict["gamma"]
    ga_log_pi = ga_logits - jax.nn.logsumexp(ga_logits, axis=-1, keepdims=True)
    ga_alpha = jnp.exp(ga_log_alpha) + 1e-4
    ga_beta  = jnp.exp(ga_log_beta) + 1e-4
    ga_comp = gamma_log_prob(t2, ga_alpha, ga_beta)           # (B, K)
    ll_t2 = jax.nn.logsumexp(ga_log_pi + ga_comp, axis=-1)   # (B,)

    # --- Beta (t3) ---
    b_logits, b_log_a, b_log_b = out_dict["beta"]
    b_log_pi = b_logits - jax.nn.logsumexp(b_logits, axis=-1, keepdims=True)
    b_a = jnp.exp(b_log_a) + 1e-4
    b_b = jnp.exp(b_log_b) + 1e-4
    b_comp = beta_log_prob(t3, b_a, b_b)                     # (B, K)
    ll_t3 = jax.nn.logsumexp(b_log_pi + b_comp, axis=-1)     # (B,)

    return -jnp.mean(ll_t1 + ll_t2 + ll_t3)


# ---------------------------------------------------------------------------
# Train step & state
# ---------------------------------------------------------------------------

@jax.jit
def train_step(state: train_state.TrainState,
               x: jnp.ndarray,
               theta: jnp.ndarray,
               ) -> Tuple[train_state.TrainState, jnp.ndarray]:

    def loss_fn(params):
        out_dict = state.apply_fn(params, x)
        return mixed_mdn_log_prob(out_dict, theta)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def create_train_state(rng, model, learning_rate, batch_size, x_dim):
    params = model.init(rng, jnp.zeros((batch_size, x_dim)))
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def train_mixed_mdn(rng, model, x_data, theta_data,
                     lr, n_epochs, batch_size, save_path):
    import os
    os.makedirs(save_path, exist_ok=True)

    n = x_data.shape[0]
    x_dim = x_data.shape[1]
    state = create_train_state(rng, model, lr, batch_size, x_dim)
    losses = []

    for ep in range(1, n_epochs + 1):
        rng, pk = random.split(rng)
        perm = random.permutation(pk, n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_data[idx], theta_data[idx]
            state, loss = train_step(state, xb, yb)
            losses.append(loss)
        if ep % 10 == 0:
            print(f"Epoch {ep:03d}  loss={loss:.4f}")

    save_multiMDN(path=save_path, params=state.params)
    return state, losses


# ---------------------------------------------------------------------------
# Extract distributional params from model output (used by CDF / inverse)
# ---------------------------------------------------------------------------

def _extract_params(model, params, x):
    """
    Returns processed parameters for each head.

    Returns
    -------
    gaussian_params : (pi, mu, sigma)         each (B, K)
    gamma_params    : (pi, alpha, beta)        each (B, K)
    beta_params     : (pi, a, b)               each (B, K)
    """
    out = model.apply(params, x)

    # Gaussian
    g_logits, g_mu, g_log_sigma = out["gaussian"]
    g_pi = jnp.exp(g_logits - jax.nn.logsumexp(g_logits, -1, keepdims=True))
    g_sigma = jnp.exp(g_log_sigma)

    # Gamma
    ga_logits, ga_log_alpha, ga_log_beta = out["gamma"]
    ga_pi = jnp.exp(ga_logits - jax.nn.logsumexp(ga_logits, -1, keepdims=True))
    ga_alpha = jnp.exp(ga_log_alpha) + 1e-4
    ga_beta  = jnp.exp(ga_log_beta) + 1e-4

    # Beta
    b_logits, b_log_a, b_log_b = out["beta"]
    b_pi = jnp.exp(b_logits - jax.nn.logsumexp(b_logits, -1, keepdims=True))
    b_a = jnp.exp(b_log_a) + 1e-4
    b_b = jnp.exp(b_log_b) + 1e-4

    return (g_pi, g_mu, g_sigma), (ga_pi, ga_alpha, ga_beta), (b_pi, b_a, b_b)


# ---------------------------------------------------------------------------
# Marginal CDF  (PIT:  theta -> u)
# ---------------------------------------------------------------------------

def get_mixed_cdf_vals(model, params, x_data, theta_data):
    """
    Compute u = F(theta | x) per marginal dimension.

    x_data:     (B, x_dim)
    theta_data: (B, 3)
    returns:    (B, 3)  with values in [0, 1]
    """
    gauss_p, gamma_p, beta_p = _extract_params(model, params, x_data)
    t1, t2, t3 = theta_data[:, 0], theta_data[:, 1], theta_data[:, 2]

    # Gaussian mixture CDF for t1
    g_pi, g_mu, g_sigma = gauss_p
    u1 = jnp.sum(g_pi * gaussian_cdf(t1, g_mu, g_sigma), axis=-1)  # (B,)

    # Gamma mixture CDF for t2
    ga_pi, ga_alpha, ga_beta = gamma_p
    u2 = jnp.sum(ga_pi * gamma_cdf(t2, ga_alpha, ga_beta), axis=-1)  # (B,)

    # Beta mixture CDF for t3
    b_pi, b_a, b_b = beta_p
    u3 = jnp.sum(b_pi * beta_cdf(t3, b_a, b_b), axis=-1)  # (B,)

    return jnp.stack([u1, u2, u3], axis=-1)  # (B, 3)


# ---------------------------------------------------------------------------
# Marginal inverse CDF  (u -> theta)   via bisection
# ---------------------------------------------------------------------------

@jax.jit
def _bisect_gamma(u, pi, alpha, beta, iters=60):
    """Batched bisection for Gamma mixture CDF inverse."""
    dtype = alpha.dtype
    u = u.astype(dtype)
    u = jnp.clip(u, jnp.array(1e-12, dtype=dtype), jnp.array(1.0 - 1e-12, dtype=dtype))
    L = jnp.array(10.0, dtype=dtype)
    ga_mean = jnp.sum(pi * alpha / beta, axis=-1)
    ga_var = jnp.sum(pi * alpha / beta**2, axis=-1)
    lo = jnp.full_like(u, 1e-12, dtype=dtype)
    hi = ga_mean + L * jnp.sqrt(ga_var + jnp.array(1e-12, dtype=dtype))

    def body(carry, _):
        lo, hi = carry
        mid = jnp.array(0.5, dtype=dtype) * (lo + hi)
        F = jnp.sum(pi * gamma_cdf(mid, alpha, beta), axis=-1)
        lo = jnp.where(F < u, mid, lo)
        hi = jnp.where(F >= u, mid, hi)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=iters)
    return jnp.array(0.5, dtype=dtype) * (lo + hi)


@jax.jit
def _bisect_beta(u, pi, a, b, iters=60):
    """Batched bisection for Beta mixture CDF inverse."""
    dtype = a.dtype
    u = u.astype(dtype)
    u = jnp.clip(u, jnp.array(1e-12, dtype=dtype), jnp.array(1.0 - 1e-12, dtype=dtype))
    lo = jnp.full_like(u, 1e-12, dtype=dtype)
    hi = jnp.full_like(u, 1.0 - 1e-12, dtype=dtype)

    def body(carry, _):
        lo, hi = carry
        mid = jnp.array(0.5, dtype=dtype) * (lo + hi)
        F = jnp.sum(pi * beta_cdf(mid, a, b), axis=-1)
        lo = jnp.where(F < u, mid, lo)
        hi = jnp.where(F >= u, mid, hi)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=iters)
    return jnp.array(0.5, dtype=dtype) * (lo + hi)


@jax.jit
def _bisect_gaussian(u, pi, mu, sigma, iters=60):
    """Batched bisection for Gaussian mixture CDF inverse."""
    dtype = mu.dtype
    u = u.astype(dtype)
    u = jnp.clip(u, jnp.array(1e-12, dtype=dtype), jnp.array(1.0 - 1e-12, dtype=dtype))
    L = jnp.array(10.0, dtype=dtype)
    m = jnp.sum(pi * mu, axis=-1)
    s = jnp.sqrt(jnp.sum(pi * (sigma**2 + mu**2), axis=-1) - m**2 + jnp.array(1e-12, dtype=dtype))
    maxsig = jnp.max(sigma, axis=-1)
    lo = m - L * (s + maxsig)
    hi = m + L * (s + maxsig)

    def body(carry, _):
        lo, hi = carry
        mid = jnp.array(0.5, dtype=dtype) * (lo + hi)
        F = jnp.sum(pi * gaussian_cdf(mid, mu, sigma), axis=-1)
        lo = jnp.where(F < u, mid, lo)
        hi = jnp.where(F >= u, mid, hi)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=iters)
    return jnp.array(0.5, dtype=dtype) * (lo + hi)



def mixed_mdn_inv_marg(model, params, x, u):
    """
    Invert the three learned marginal CDFs.
    x: (..., x_dim)
    u: (..., 3)
    returns theta: (..., 3)
    """
    gauss_p, gamma_p, beta_p = _extract_params(model, params, x)

    g_pi, g_mu, g_sigma = gauss_p
    ga_pi, ga_alpha, ga_beta = gamma_p
    b_pi, b_a, b_b = beta_p

    t1 = _bisect_gaussian(u[..., 0], g_pi, g_mu, g_sigma)
    t2 = _bisect_gamma(u[..., 1], ga_pi, ga_alpha, ga_beta)
    t3 = _bisect_beta(u[..., 2], b_pi, b_a, b_b)

    return jnp.stack([t1, t2, t3], axis=-1)

def plot_abc_vs_mdn(model, params, key, x_data, 
                    n_obs=4, n_accept=5000, eps_quantile=0.002,
                    batch_size=2_000_000, save_path="./"):
    """
    Pick n_obs observations from x_data, run ABC to get reference posteriors,
    overlay the learned MDN marginal densities, and save the figure.

    Parameters
    ----------
    model      : MixedMarginalMDN instance
    params     : trained parameter dict
    key        : PRNG key
    x_data     : (N, 1) training/validation x data to pick test points from
    n_obs      : number of test observations
    n_accept   : ABC samples per observation
    eps_quantile : quantile for auto-calibrating ABC epsilon
    batch_size : ABC simulator batch size
    save_path  : directory to save the figure
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gamma as sp_gamma, beta as sp_beta
    from abc_reference import abc_rejection_multi, calibrate_eps

    os.makedirs(save_path, exist_ok=True)

    k1, k2, k3 = random.split(key, 3)

    # pick test observations
    test_ids = random.choice(k1, x_data.shape[0], (n_obs,), replace=False)
    x_test = x_data[test_ids]  # (n_obs, 1)

    # calibrate epsilon and run ABC
    eps = calibrate_eps(k2, quantile=eps_quantile, pilot_n=batch_size)

    theta_abc = abc_rejection_multi(
        k3, x_test.squeeze(),
        n_accept=n_accept, eps=eps,
        batch_size=batch_size,
    )  # (n_obs, n_accept, 3)

    # extract learned MDN parameters
    gauss_p, gamma_p, beta_p = _extract_params(model, params, x_test)

    names = [r"$\theta_1$ (Normal)", r"$\theta_2$ (Gamma)", r"$\theta_3$ (Beta)"]

    fig, axes = plt.subplots(n_obs, 3, figsize=(14, 3.5 * n_obs))
    if n_obs == 1:
        axes = axes[None, :]

    for i in range(n_obs):
        x_val = float(x_test[i, 0])

        # --- t1: Gaussian mixture ---
        ax = axes[i, 0]
        samples = np.array(theta_abc[i, :, 0])
        ax.hist(samples, bins=60, density=True, alpha=0.4, color="C0", label="ABC")
        t_grid = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 300)
        pi, mu, sigma = gauss_p[0][i], gauss_p[1][i], gauss_p[2][i]
        pdf = sum(
            float(pi[k]) * np.exp(-0.5 * ((t_grid - float(mu[k])) / float(sigma[k]))**2)
            / (float(sigma[k]) * np.sqrt(2 * np.pi))
            for k in range(pi.shape[0])
        )
        ax.plot(t_grid, pdf, "C1-", lw=2, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[0]}")
        ax.legend(fontsize=8)

        # --- t2: Gamma mixture ---
        ax = axes[i, 1]
        samples = np.array(theta_abc[i, :, 1])
        ax.hist(samples, bins=60, density=True, alpha=0.4, color="C0", label="ABC")
        t_grid = np.linspace(max(1e-6, samples.min() - 0.3), samples.max() + 0.5, 300)
        pi, alpha, beta = gamma_p[0][i], gamma_p[1][i], gamma_p[2][i]
        pdf = sum(
            float(pi[k]) * sp_gamma.pdf(t_grid, a=float(alpha[k]), scale=1.0/float(beta[k]))
            for k in range(pi.shape[0])
        )
        ax.plot(t_grid, pdf, "C1-", lw=2, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[1]}")
        ax.legend(fontsize=8)

        # --- t3: Beta mixture ---
        ax = axes[i, 2]
        samples = np.array(theta_abc[i, :, 2])
        ax.hist(samples, bins=60, density=True, alpha=0.4, color="C0", label="ABC")
        t_grid = np.linspace(1e-4, 1.0 - 1e-4, 300)
        pi, a, b = beta_p[0][i], beta_p[1][i], beta_p[2][i]
        pdf = sum(
            float(pi[k]) * sp_beta.pdf(t_grid, float(a[k]), float(b[k]))
            for k in range(pi.shape[0])
        )
        ax.plot(t_grid, pdf, "C1-", lw=2, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[2]}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abc_vs_mdn.pdf"))
    plt.close()
    print(f"Saved abc_vs_mdn.pdf to {save_path}")




# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data import gen_mixed_prior_binomial_normal_data
    from abc_reference import abc_rejection_multi, calibrate_eps
    from plots import plot_loss
    import matplotlib.pyplot as plt
    import numpy as np

    key = random.PRNGKey(42)
    k1, k2, k3, k4 = random.split(key, 4)

    x_data, theta_data = gen_mixed_prior_binomial_normal_data(k1, 50_000)
    print("x:", x_data.shape, "theta:", theta_data.shape)

    model = MixedMarginalMDN(hidden_dims=[64, 64], K=5)
    state, losses = train_mixed_mdn(k2, model, x_data, theta_data,
                                    lr=1e-3, n_epochs=200,
                                    batch_size=5000,
                                    save_path="mixed_mdn_test/")
    plot_loss(losses, "mixed_mdn_test/")

    # --- PIT check ---
    u = get_mixed_cdf_vals(model, state.params, x_data[:1000], theta_data[:1000])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for j, name in enumerate(["t1 (Gauss)", "t2 (Gamma)", "t3 (Beta)"]):
        axes[j].hist(np.array(u[:, j]), bins=30, density=True)
        axes[j].set_title(f"PIT {name}")
        axes[j].axhline(1.0, color="r", ls="--")
    plt.tight_layout()
    plt.savefig("mixed_mdn_test/pit_check.pdf")
    plt.close()

    # --- Round-trip check ---
    theta_rt = mixed_mdn_inv_marg(model, state.params, x_data[:1000], u)
    err = jnp.abs(theta_rt - theta_data[:1000])
    print("Round-trip max abs error per dim:", err.max(axis=0))

    # --- ABC reference posterior + learned MDN overlay ---
    n_obs = 4
    n_accept = 5000

    # pick a few test observations from the data
    test_ids = random.choice(k3, x_data.shape[0], (n_obs,), replace=False)
    x_test = x_data[test_ids]       # (n_obs, 1)
    theta_test = theta_data[test_ids]  # (n_obs, 3)

    # auto-calibrate epsilon
    eps = calibrate_eps(k4, quantile=0.002)

    # run ABC
    key, ak = random.split(k4)
    theta_abc = abc_rejection_multi(
        ak, x_test.squeeze(),
        n_accept=n_accept, eps=eps,
        batch_size=2_000_000,
    )  # (n_obs, n_accept, 3)

    # extract learned MDN parameters at test locations
    gauss_p, gamma_p, beta_p = _extract_params(model, state.params, x_test)

    names = [r"$\theta_1$ (Normal)", r"$\theta_2$ (Gamma)", r"$\theta_3$ (Beta)"]

    fig, axes = plt.subplots(n_obs, 3, figsize=(14, 3.5 * n_obs))
    if n_obs == 1:
        axes = axes[None, :]

    for i in range(n_obs):
        x_val = float(x_test[i, 0])

        # --- t1: Gaussian mixture ---
        ax = axes[i, 0]
        samples = np.array(theta_abc[i, :, 0])
        ax.hist(samples, bins=60, density=True, alpha=0.4, color="C0", label="ABC")
        t_grid = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 300)
        pi, mu, sigma = gauss_p[0][i], gauss_p[1][i], gauss_p[2][i]  # each (K,)
        pdf = sum(
            float(pi[k]) * np.exp(-0.5 * ((t_grid - float(mu[k])) / float(sigma[k]))**2)
            / (float(sigma[k]) * np.sqrt(2 * np.pi))
            for k in range(pi.shape[0])
        )
        ax.plot(t_grid, pdf, "C1-", lw=2, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[0]}")
        ax.legend(fontsize=8)

        # --- t2: Gamma mixture ---
        ax = axes[i, 1]
        samples = np.array(theta_abc[i, :, 1])
        ax.hist(samples, bins=60, density=True, alpha=0.4, color="C0", label="ABC")
        t_grid = np.linspace(max(1e-6, samples.min() - 0.3), samples.max() + 0.5, 300)
        pi, alpha, beta = gamma_p[0][i], gamma_p[1][i], gamma_p[2][i]
        from scipy.stats import gamma as sp_gamma
        pdf = sum(
            float(pi[k]) * sp_gamma.pdf(t_grid, a=float(alpha[k]), scale=1.0 / float(beta[k]))
            for k in range(pi.shape[0])
        )
        ax.plot(t_grid, pdf, "C1-", lw=2, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[1]}")
        ax.legend(fontsize=8)

        # --- t3: Beta mixture ---
        ax = axes[i, 2]
        samples = np.array(theta_abc[i, :, 2])
        ax.hist(samples, bins=60, density=True, alpha=0.4, color="C0", label="ABC")
        t_grid = np.linspace(1e-4, 1.0 - 1e-4, 300)
        pi, a, b = beta_p[0][i], beta_p[1][i], beta_p[2][i]
        from scipy.stats import beta as sp_beta
        pdf = sum(
            float(pi[k]) * sp_beta.pdf(t_grid, float(a[k]), float(b[k]))
            for k in range(pi.shape[0])
        )
        ax.plot(t_grid, pdf, "C1-", lw=2, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[2]}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("mixed_mdn_test/abc_vs_mdn.pdf")
    plt.show()
    plt.close()

    print("Saved abc_vs_mdn.pdf")

