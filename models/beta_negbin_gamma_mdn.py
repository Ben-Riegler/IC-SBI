import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
from typing import List, Tuple
import time
from utils import standardize_cols

from utils import save_multiMDN

jax.config.update("jax_enable_x64", False)


# ===========================================================================
# Data Generating Process
# ===========================================================================

def gen_beta_negbin_gamma_data(key, n_samples):
    """
    DGP:
        t1 ~ Beta(2, 3)
        t2 ~ NegBin(r=20, p=t1)
        t3 ~ Gamma(2, t1)             [shape=t1, rate=2]
        c  ~ Binomial(t2, t1)
        z  ~ N(0, t3)                 [t3 is variance]
        x  = c + z

    Returns
    -------
    x_data     : (n_samples, 1)
    theta_data : (n_samples, 3)   columns are [t1, t2, t3]
    """
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)

    # prior draws
    t1 = random.beta(k1, 2.0, 2.0, shape=(n_samples,))  # Beta(2,3)

    # t2 ~ NegBin(r=20, p=t1) via Gamma-Poisson mixture
    r = 3.0
    gamma_nb = random.gamma(k2, r, shape=(n_samples,))  # Gamma(shape=r, scale=1)
    gamma_nb = gamma_nb * (1.0 - t1) / t1  # scale to (1-p)/p
    t2 = random.poisson(k3, gamma_nb).astype(jnp.float32)

    # t3 ~ Gamma(shape=2, rate=t1)  →  scale = 1/2
    # JAX gamma samples Gamma(shape, scale=1), so multiply by 1/rate
    t3 = random.gamma(k4, 1.0, shape=(n_samples,)) / (10*t1)

    # simulator
    c = random.binomial(k5, n=t2.astype(jnp.int32), p=t1).astype(jnp.float32)
    z = jnp.sqrt(t3) * random.normal(k6, (n_samples,))  # N(0, t3)
    x = c + z

    theta_data = jnp.stack([t1, t2, t3], axis=-1)  # (n_samples, 3)
    x_data = x[:, None]  # (n_samples, 1)

    return x_data, theta_data


# ===========================================================================
# Log-pdf helpers
# ===========================================================================

def beta_log_prob(y, a, b):
    """Beta(y; a, b).  y: (...,)  a,b: (..., K)  returns: (..., K)"""
    y_k = jnp.clip(y[..., None], 1e-12, 1.0 - 1e-12)
    return (
        (a - 1.0) * jnp.log(y_k)
        + (b - 1.0) * jnp.log(1.0 - y_k)
        + jax.lax.lgamma(a + b)
        - jax.lax.lgamma(a)
        - jax.lax.lgamma(b)
    )


def negbin_log_prob(y, r, logit_p):
    """
    NegBin(y; r, p) — number of failures before r successes.
    y: (...,)  r: (..., K)  logit_p: (..., K)  returns: (..., K)
    """
    y_k = y[..., None]
    log_p = jax.nn.log_sigmoid(logit_p)
    log_1mp = jax.nn.log_sigmoid(-logit_p)
    return (
        jax.lax.lgamma(y_k + r)
        - jax.lax.lgamma(y_k + 1.0)
        - jax.lax.lgamma(r)
        + r * log_p
        + y_k * log_1mp
    )


def gamma_log_prob(y, alpha, beta):
    """
    Gamma(y; alpha, beta) with shape alpha, rate beta.
    y: (...,)  alpha,beta: (..., K)  returns: (..., K)
    """
    y_k = jnp.clip(y[..., None], 1e-12)
    return (
        alpha * jnp.log(beta)
        - jax.lax.lgamma(alpha)
        + (alpha - 1.0) * jnp.log(y_k)
        - beta * y_k
    )


# ===========================================================================
# CDF helpers
# ===========================================================================

def beta_cdf(y, a, b):
    """Component CDFs for Beta(a, b). returns (..., K)"""
    y_k = jnp.clip(y[..., None], 1e-12, 1.0 - 1e-12)
    y_k = y_k.astype(a.dtype)
    return jax.lax.betainc(a, b, y_k)


def negbin_cdf(k, r, p):
    """
    CDF of NegBin(r, p) at integer k.
    P(X ≤ k) = I_p(r, k+1) = betainc(r, k+1, p)
    k: (...,) or (..., 1)   r,p: (..., K)   returns: (..., K)
    """
    k_val = k if k.ndim == r.ndim else k[..., None]
    k_safe = jnp.maximum(k_val, 0.0)
    cdf_val = jax.lax.betainc(r, k_safe.astype(r.dtype) + 1.0, p)
    cdf_val = jnp.where(k_val < 0, 0.0, cdf_val)
    return cdf_val


def gamma_cdf(y, alpha, beta):
    """Component CDFs for Gamma(alpha, beta). returns (..., K)"""
    y_k = jnp.clip(y[..., None], 1e-30)
    y_k = y_k.astype(alpha.dtype)
    return jax.lax.igamma(alpha, beta * y_k)


def negbin_mixture_cdf(k, pis, r, p):
    """CDF of NegBin mixture.  k: (...,)  pis,r,p: (..., K)  returns: (...,)"""
    comp_cdfs = negbin_cdf(k[..., None], r, p)
    return jnp.sum(pis * comp_cdfs, axis=-1)


# ===========================================================================
# Randomized PIT for NegBin mixture
# ===========================================================================

def negbin_mixture_randomized_pit(key, theta_j, pis, r, p):
    """
    u = F(θ_j - 1) + α·[F(θ_j) - F(θ_j - 1)],  α ~ U(0,1)
    theta_j: (...,)  pis,r,p: (..., K)  returns: (...,)
    """
    alpha = random.uniform(key, theta_j.shape, dtype=pis.dtype)
    F_theta = negbin_mixture_cdf(theta_j, pis, r, p)
    F_theta_m1 = negbin_mixture_cdf(theta_j - 1.0, pis, r, p)
    u = F_theta_m1 + alpha * (F_theta - F_theta_m1)
    return jnp.clip(u, 1e-12, 1.0 - 1e-12)


# ===========================================================================
# Inverse CDFs (bisection)
# ===========================================================================

@jax.jit
def _bisect_beta(u, pi, a, b, iters=60):
    """Batched bisection for Beta mixture CDF inverse."""
    dtype = a.dtype
    u = u.astype(dtype)
    u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
    lo = jnp.full_like(u, 1e-12)
    hi = jnp.full_like(u, 1.0 - 1e-12)

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        F = jnp.sum(pi * beta_cdf(mid, a, b), axis=-1)
        lo = jnp.where(F < u, mid, lo)
        hi = jnp.where(F >= u, mid, hi)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=iters)
    return 0.5 * (lo + hi)


@jax.jit
def _bisect_negbin(u, pis, r, p, max_val=2000, iters=30):
    """Integer bisection for NegBin mixture inverse CDF."""
    u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
    lo = jnp.zeros_like(u)
    hi = jnp.full_like(u, float(max_val))

    def body(carry, _):
        lo, hi = carry
        mid = jnp.floor(0.5 * (lo + hi))
        F = negbin_mixture_cdf(mid, pis, r, p)
        lo = jnp.where(F < u, mid + 1.0, lo)
        hi = jnp.where(F >= u, mid, hi)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=iters)
    return lo


@jax.jit
def _bisect_gamma(u, pi, alpha, beta, iters=60):
    """Batched bisection for Gamma mixture CDF inverse."""
    dtype = alpha.dtype
    u = u.astype(dtype)
    u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
    ga_mean = jnp.sum(pi * alpha / beta, axis=-1)
    ga_var = jnp.sum(pi * alpha / beta**2, axis=-1)
    lo = jnp.full_like(u, 1e-12)
    hi = ga_mean + 10.0 * jnp.sqrt(ga_var + 1e-12)
    hi = jnp.maximum(hi, 500.0)

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        F = jnp.sum(pi * gamma_cdf(mid, alpha, beta), axis=-1)
        lo = jnp.where(F < u, mid, lo)
        hi = jnp.where(F >= u, mid, hi)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=iters)
    return 0.5 * (lo + hi)


# ===========================================================================
# Model
# ===========================================================================

class BetaNegBinGammaMDN(nn.Module):
    """
    Shared encoder → three independent mixture heads:
        dim 0 (t1): mixture of Betas     → (logits, log_a, log_b)
        dim 1 (t2): mixture of NegBins   → (logits, log_r, logit_p)
        dim 2 (t3): mixture of Gammas    → (logits, log_alpha, log_beta)
    """
    hidden_dims: List[int]
    K: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        h = x
        for dim in self.hidden_dims[:-1]:
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
        h = nn.Dense(self.hidden_dims[-1])(h)
        h = nn.relu(h)

        # Beta head (t1)
        b_logits = nn.Dense(self.K, name="b_logits")(h)
        b_log_a = nn.Dense(self.K, name="b_log_a")(h)
        b_log_b = nn.Dense(self.K, name="b_log_b")(h)

        # NegBin head (t2)
        nb_logits = nn.Dense(self.K, name="nb_logits")(h)
        nb_log_r = nn.Dense(self.K, name="nb_log_r")(h)
        nb_logit_p = nn.Dense(self.K, name="nb_logit_p")(h)

        # Gamma head (t3)
        ga_logits = nn.Dense(self.K, name="ga_logits")(h)
        ga_log_alpha = nn.Dense(self.K, name="ga_log_alpha")(h)
        ga_log_beta = nn.Dense(self.K, name="ga_log_beta")(h)

        return {
            "beta": (b_logits, b_log_a, b_log_b),
            "negbin": (nb_logits, nb_log_r, nb_logit_p),
            "gamma": (ga_logits, ga_log_alpha, ga_log_beta),
        }


# ===========================================================================
# Loss
# ===========================================================================

def mdn_log_prob(out_dict, theta):
    """
    theta: (B, 3) — columns [t1 (Beta), t2 (NegBin), t3 (Gamma)]
    returns: scalar (mean negative log-likelihood)
    """
    t1, t2, t3 = theta[:, 0], theta[:, 1], theta[:, 2]

    # Beta (t1)
    b_logits, b_log_a, b_log_b = out_dict["beta"]
    b_log_pi = b_logits - jax.nn.logsumexp(b_logits, axis=-1, keepdims=True)
    b_a = jnp.exp(b_log_a) + 1e-4
    b_b = jnp.exp(b_log_b) + 1e-4
    ll_t1 = jax.nn.logsumexp(b_log_pi + beta_log_prob(t1, b_a, b_b), axis=-1)

    # NegBin (t2)
    nb_logits, nb_log_r, nb_logit_p = out_dict["negbin"]
    nb_log_pi = nb_logits - jax.nn.logsumexp(nb_logits, axis=-1, keepdims=True)
    nb_r = jnp.exp(nb_log_r) + 1e-4
    ll_t2 = jax.nn.logsumexp(nb_log_pi + negbin_log_prob(t2, nb_r, nb_logit_p), axis=-1)

    # Gamma (t3)
    ga_logits, ga_log_alpha, ga_log_beta = out_dict["gamma"]
    ga_log_pi = ga_logits - jax.nn.logsumexp(ga_logits, axis=-1, keepdims=True)
    ga_alpha = jnp.exp(ga_log_alpha) + 1e-4
    ga_beta = jnp.exp(ga_log_beta) + 1e-4
    ll_t3 = jax.nn.logsumexp(ga_log_pi + gamma_log_prob(t3, ga_alpha, ga_beta), axis=-1)

    return -jnp.mean(ll_t1 + ll_t2 + ll_t3)


# ===========================================================================
# Train step & state
# ===========================================================================

@jax.jit
def train_step(state, x, theta):
    def loss_fn(params):
        out_dict = state.apply_fn(params, x)
        return mdn_log_prob(out_dict, theta)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def create_train_state(rng, model, learning_rate, batch_size, x_dim):
    params = model.init(rng, jnp.zeros((batch_size, x_dim)))
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )


def train_mdn(rng, model, x_data, theta_data,
              lr, n_epochs, batch_size, save_path):
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
            state, loss = train_step(state, x_data[idx], theta_data[idx])
            losses.append(loss)
        if ep % 10 == 0:
            print(f"Epoch {ep:03d}  loss={loss:.4f}")

    save_multiMDN(path=save_path, params=state.params)
    return state, losses


# ===========================================================================
# Extract distributional params
# ===========================================================================

def _extract_params(model, params, x):
    """
    Returns
    -------
    beta_params  : (pi, a, b)       each (..., K)
    negbin_params: (pi, r, p)       each (..., K)
    gamma_params : (pi, alpha, beta) each (..., K)
    """
    out = model.apply(params, x)

    # Beta
    b_logits, b_log_a, b_log_b = out["beta"]
    b_pi = jnp.exp(b_logits - jax.nn.logsumexp(b_logits, -1, keepdims=True))
    b_a = jnp.exp(b_log_a) + 1e-4
    b_b = jnp.exp(b_log_b) + 1e-4

    # NegBin
    nb_logits, nb_log_r, nb_logit_p = out["negbin"]
    nb_pi = jnp.exp(nb_logits - jax.nn.logsumexp(nb_logits, -1, keepdims=True))
    nb_r = jnp.exp(nb_log_r) + 1e-4
    nb_p = jax.nn.sigmoid(nb_logit_p)

    # Gamma
    ga_logits, ga_log_alpha, ga_log_beta = out["gamma"]
    ga_pi = jnp.exp(ga_logits - jax.nn.logsumexp(ga_logits, -1, keepdims=True))
    ga_alpha = jnp.exp(ga_log_alpha) + 1e-4
    ga_beta = jnp.exp(ga_log_beta) + 1e-4

    return (b_pi, b_a, b_b), (nb_pi, nb_r, nb_p), (ga_pi, ga_alpha, ga_beta)


# ===========================================================================
# Marginal CDF:  theta -> u
# ===========================================================================

def get_cdf_vals(key, model, params, x_data, theta_data):
    """
    Standard PIT for Beta and Gamma (continuous).
    Randomized PIT for NegBin (discrete).

    x_data:     (B, x_dim)
    theta_data: (B, 3)
    returns:    (B, 3)
    """
    beta_p, negbin_p, gamma_p = _extract_params(model, params, x_data)
    t1, t2, t3 = theta_data[:, 0], theta_data[:, 1], theta_data[:, 2]

    # Beta mixture CDF for t1 (continuous)
    b_pi, b_a, b_b = beta_p
    u1 = jnp.sum(b_pi * beta_cdf(t1, b_a, b_b), axis=-1)

    # NegBin mixture for t2 (discrete → randomized PIT)
    nb_pi, nb_r, nb_p = negbin_p
    u2 = negbin_mixture_randomized_pit(key, t2, nb_pi, nb_r, nb_p)

    # Gamma mixture CDF for t3 (continuous)
    ga_pi, ga_alpha, ga_beta = gamma_p
    u3 = jnp.sum(ga_pi * gamma_cdf(t3, ga_alpha, ga_beta), axis=-1)

    return jnp.stack([u1, u2, u3], axis=-1)


# ===========================================================================
# Marginal inverse CDF:  u -> theta
# ===========================================================================

def inv_marginals(model, params, x, u):
    """
    Invert the three learned marginal CDFs.
    x: (..., x_dim)
    u: (..., 3)
    returns theta: (..., 3)
    """
    beta_p, negbin_p, gamma_p = _extract_params(model, params, x)

    b_pi, b_a, b_b = beta_p
    nb_pi, nb_r, nb_p = negbin_p
    ga_pi, ga_alpha, ga_beta = gamma_p

    t1 = _bisect_beta(u[..., 0], b_pi, b_a, b_b)
    t2 = _bisect_negbin(u[..., 1], nb_pi, nb_r, nb_p)
    t3 = _bisect_gamma(u[..., 2], ga_pi, ga_alpha, ga_beta)

    return jnp.stack([t1, t2, t3], axis=-1)


# ===========================================================================
# ABC reference
# ===========================================================================

def abc_rejection(key, x_obs, mean, std,n_accept=5000,
                  eps=0.01, batch_size=2_000_000):
    """
    ABC rejection sampler for the Beta-NegBin-Gamma DGP.
    Vectorized over all observations simultaneously.

    x_obs: (n_obs,)
    returns: (n_obs, n_accept, 3)
    """
    n_obs = x_obs.shape[0]
    accepted = [[] for _ in range(n_obs)]
    counts = jnp.zeros(n_obs, dtype=jnp.int32)

    while int(counts.min()) < n_accept:
        key, k1 = random.split(key)
        x_sim, theta_batch = gen_beta_negbin_gamma_data(k1, batch_size)
        x_sim = standardize_cols(x_sim, mean, std)[0]

        # accept/reject for all observations
        dists = jnp.abs(x_sim - x_obs[None, :])
 

        for i in range(n_obs):
            if int(counts[i]) >= n_accept:
                continue
            mask = dists[:, i] < eps
            acc = theta_batch[mask]
            if acc.shape[0] > 0:
                accepted[i].append(acc)
                counts = counts.at[i].set(counts[i] + acc.shape[0])

        done = int(counts.min())
        print(f"\r  ABC: min accepted = {done}/{n_accept}", end="", flush=True)

    print()

    results = []
    for i in range(n_obs):
        cat = jnp.concatenate(accepted[i], axis=0)[:n_accept]
        results.append(cat)

    return jnp.stack(results, axis=0)


# ===========================================================================
# Plotting: ABC reference vs MDN
# ===========================================================================

def plot_abc_vs_mdn(model, params, key, x_data, mean, std,
                    n_obs=4, n_accept=5000, eps=1.0, save_path="./"):
    """
    Pick n_obs observations, run ABC, overlay with learned MDN.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import beta as sp_beta, nbinom as sp_nbinom, gamma as sp_gamma

    os.makedirs(save_path, exist_ok=True)

    k1, k2 = random.split(key)
    test_ids = random.choice(k1, x_data.shape[0], (n_obs,), replace=False)
    x_test = x_data[test_ids]

    # Run ABC
    theta_abc = abc_rejection(k2, x_test.squeeze(), mean, std, n_accept=n_accept, eps=eps)
 
    # Extract MDN params
    beta_p, negbin_p, gamma_p = _extract_params(model, params, x_test)

    names = [r"$\theta_1$ (Beta)", r"$\theta_2$ (NegBin)", r"$\theta_3$ (Gamma)"]

    fig, axes = plt.subplots(n_obs, 3, figsize=(14, 3.5 * n_obs))
    if n_obs == 1:
        axes = axes[None, :]

    for i in range(n_obs):
        x_val = float(x_test[i, 0])

        # --- t1: Beta ---
        ax = axes[i, 0]
        abc_t1 = np.array(theta_abc[i, :, 0])
        ax.hist(abc_t1, bins=60, density=True, alpha=0.4, color="C0", label="ABC")
        t_grid = np.linspace(1e-4, 1.0 - 1e-4, 300)
        pi, a, b = beta_p[0][i], beta_p[1][i], beta_p[2][i]
        pdf = sum(
            float(pi[k]) * sp_beta.pdf(t_grid, float(a[k]), float(b[k]))
            for k in range(pi.shape[0])
        )
        ax.plot(t_grid, pdf, "C1-", lw=2, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[0]}")
        ax.legend(fontsize=8)

        # --- t2: NegBin ---
        ax = axes[i, 1]
        abc_t2 = np.array(theta_abc[i, :, 1])
        k_max = max(int(abc_t2.max()) + 10, 50)
        k_grid = np.arange(0, k_max)
        ax.hist(abc_t2, bins=k_grid - 0.5, density=True, alpha=0.4,
                color="C0", label="ABC")

        nb_pi, nb_r, nb_p = negbin_p[0][i], negbin_p[1][i], negbin_p[2][i]
        pmf = np.zeros(k_max)
        for k_comp in range(nb_pi.shape[0]):
            pmf += float(nb_pi[k_comp]) * sp_nbinom.pmf(
                k_grid, n=float(nb_r[k_comp]), p=float(nb_p[k_comp])
            )
        ax.bar(k_grid, pmf, alpha=0.6, color="C1", width=0.8, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[1]}")
        ax.legend(fontsize=8)

        # --- t3: Gamma ---
        ax = axes[i, 2]
        abc_t3 = np.array(theta_abc[i, :, 2])
        ax.hist(abc_t3, bins=60, density=True, alpha=0.4, color="C0", label="ABC")
        t_grid = np.linspace(max(1e-6, abc_t3.min() - 0.1),
                             abc_t3.max() + 0.5, 300)
        ga_pi, ga_alpha, ga_beta = gamma_p[0][i], gamma_p[1][i], gamma_p[2][i]
        pdf = sum(
            float(ga_pi[k]) * sp_gamma.pdf(
                t_grid, a=float(ga_alpha[k]), scale=1.0 / float(ga_beta[k])
            )
            for k in range(ga_pi.shape[0])
        )
        ax.plot(t_grid, pdf, "C1-", lw=2, label="MDN")
        ax.set_title(f"x={x_val:.2f}  {names[2]}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "abc_vs_mdn.pdf"))
    plt.close()
    print(f"Saved abc_vs_mdn.pdf to {save_path}")


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    from plots import plot_loss
    import matplotlib.pyplot as plt
    import numpy as np


    key = random.PRNGKey(42)
    k1, k2, k3, k4 = random.split(key, 4)

    # generate data
    x_data, theta_data = gen_beta_negbin_gamma_data(k1, 10000)
    x_data, mean, std = standardize_cols(x_data)
    print("x:", x_data.shape, "theta:", theta_data.shape)
    print("t1 (Beta)   range:", float(theta_data[:, 0].min()), float(theta_data[:, 0].max()))
    print("t2 (NegBin) range:", float(theta_data[:, 1].min()), float(theta_data[:, 1].max()))
    print("t3 (Gamma)  range:", float(theta_data[:, 2].min()), float(theta_data[:, 2].max()))

    # train
    model = BetaNegBinGammaMDN(hidden_dims=[32, 32], K=2)
    state, losses = train_mdn(
        k2, model, x_data, theta_data,
        lr=1e-3, n_epochs=2000, batch_size=10000,
        save_path="beta_negbin_gamma_mdn_test/",
    )
    plot_loss(losses, "beta_negbin_gamma_mdn_test/")

    # PIT check
    u = get_cdf_vals(k3, model, state.params, x_data[:2000], theta_data[:2000])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for j, name in enumerate(["t1 (Beta, std PIT)", "t2 (NegBin, rand PIT)", "t3 (Gamma, std PIT)"]):
        axes[j].hist(np.array(u[:, j]), bins=30, density=True)
        axes[j].set_title(f"PIT {name}")
        axes[j].axhline(1.0, color="r", ls="--")
    plt.tight_layout()
    plt.savefig("beta_negbin_gamma_mdn_test/pit_check.pdf")
    plt.close()

    # Round-trip check
    theta_rt = inv_marginals(model, state.params, x_data[:500], u[:500])
    err_t1 = jnp.abs(theta_rt[:, 0] - theta_data[:500, 0])
    err_t2 = jnp.abs(theta_rt[:, 1] - theta_data[:500, 1])
    err_t3 = jnp.abs(theta_rt[:, 2] - theta_data[:500, 2])
    print(f"Round-trip t1 (Beta)   max abs error: {err_t1.max():.6f}")
    print(f"Round-trip t2 (NegBin) max abs error: {err_t2.max():.1f}")
    print(f"Round-trip t3 (Gamma)  max abs error: {err_t3.max():.6f}")

    # ABC vs MDN plot
    plot_abc_vs_mdn(
        model, state.params, k4,
        x_data, mean, std,
        n_obs=4, n_accept=10000, eps=0.01,
        save_path="beta_negbin_gamma_mdn_test/",
    )

    print("Done!")
