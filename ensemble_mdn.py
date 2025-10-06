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
import matplotlib.pyplot as plt
import time

from plots import plot_mvn_data
from mdn import MDN


def gen_mv_normal_normal_data(key,
                              n_samples,
                              prior_mean,
                              prior_L,
                              model_L):
    k1, k2 = random.split(key) 
    d = prior_L.shape[0]
    ε = random.normal(k1, (n_samples, d, 1))
    θ =  prior_mean + jnp.matmul(prior_L, ε )
    ν = random.normal(k2, (n_samples, d, 1))
    x = θ + jnp.matmul(model_L, ν)

    LL0 = jnp.matmul(prior_L, prior_L.T)
    LL1 = jnp.matmul(model_L, model_L.T) 
    Varx = LL0 + LL1
    Varx_inv = jnp.linalg.inv(Varx)
    Covθx = LL0
    B = jnp.linalg.matmul(Covθx, Varx_inv)
    post_mean = prior_mean + jnp.linalg.matmul(B, x-prior_mean)
    post_var = LL0 - jnp.linalg.matmul(B, Covθx)

    return x.squeeze(-1), θ.squeeze(-1), post_mean.squeeze(-1), post_var #[None, :].repeat(repeats=n_samples, axis=0)

# d=4

# key = random.PRNGKey(1)
# k1, k2, k3 = random.split(key, 3)
# L0 = random.normal(k1, (d,d))
# L1 = random.normal(k2, (d,d))

# prior_mean = jnp.zeros((d,1))

# x, θ, post_mean, post_var = gen_mv_normal_normal_data(key, 
#                                                       n_samples=10000, 
#                                                       prior_mean=prior_mean,
#                                                       prior_L=L0, 
#                                                       model_L=L1)

# print(x.shape, θ.shape, post_mean.shape, post_var.shape)

# fig_x, fig_theta = plot_mvn_data(x, θ, feature_names=[f"dim{i}" for i in range(d)])
# plt.show()


def build_vmapped_mdn(d: int, hidden: List[int], K: int):
    """
    Returns a module that replicates MDN d times with independent parameters.
    Output shapes: (batch, d, K) for each head’s output.
    """
    # Vectorize MDN over a new axis of size d, duplicating params per axis element
    VMapped = nn.vmap(
        MDN,
        variable_axes={"params": 0},   # separate params per copy
        split_rngs={"params": True},   # different init per copy
        in_axes=None,                  # same x goes to every copy
        out_axes=1,                    # stack outputs dim axis (1)
        axis_size=d
    )

    class MultiMDN(nn.Module):
        hidden_dims: List[int]
        K: int

        @nn.compact
        def __call__(self, x: jnp.ndarray):
            logits, means, log_scales = VMapped(self.hidden_dims, self.K)(x) # [batch, d, K]

            return logits, means, log_scales

    return MultiMDN(hidden, K)

@jax.jit
def train_step(state: train_state.TrainState,
               x: jnp.ndarray, # (batch, d)
               y: jnp.ndarray # (batch, d)
              ) -> Tuple[train_state.TrainState, jnp.ndarray]:

    def loss_fn(params):
        logits, means, log_scales = state.apply_fn(params, x) # (batch, d, K)
        log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True) # (batch, d, K)
        scales = jnp.exp(log_scales) # (batch, d, K)

        log_probs = (
            -0.5 * ((y[..., None] - means)/scales)**2
            - log_scales
            - 0.5 * jnp.log(2*jnp.pi)
        ) # (batch, d, K)
        log_lik = jax.nn.logsumexp(log_pi + log_probs, axis=-1) # (batch, d)
        log_lik = jnp.sum(log_lik, axis=-1) # (batch)
        return -jnp.mean(log_lik)

    grads = jax.grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    loss  = loss_fn(state.params)
    return state, loss

def create_train_state(rng: Any, model,
                       learning_rate: float,
                       x_shape: Tuple[int,int]
                      ) -> train_state.TrainState:
    
    """Initial training state, will be updated in `train_step`"""

    params = model.init(rng, jnp.zeros(x_shape)) # model is stateless, not affected by calling init
    tx     = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def plot_losses(losses):

    fig, ax = plt.subplots(1,1)

    n = len(losses)
    ns = [i for i in range(n)]

    ax.plot(ns, losses, color="red", linewidth=0.5)
    ax.set_title("Training loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")

    fig.show()


def train_multi_mdn(rng, model, x_data, θ_data,
              lr, n_epochs, batch_size):
    state = create_train_state(rng, model, lr, x_shape=x_data.shape)
    n = x_data.shape[0]

    losses = []

    for ep in range(1, n_epochs+1):
        rng, pk = random.split(rng)
        perm = random.permutation(pk, n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = x_data[idx], θ_data[idx]
            state, loss = train_step(state, xb, yb)
            losses.append(loss)
        if ep % 10 == 0:
            print(f"Epoch {ep:03d}  loss={loss:.4f}")

    plot_losses(losses)
    return state

# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    # 1) generate data
    d=3

    key = random.PRNGKey(1)
    key, k1, k2, k3 = random.split(key, 4)
    L0 = random.normal(k1, (d,d))
    L1 = random.normal(k2, (d,d))

    prior_mean = jnp.zeros((d,1))

    x_data, θ_data, post_mean, post_var = gen_mv_normal_normal_data(key, 
                                                        n_samples=5000, 
                                                        prior_mean=prior_mean,
                                                        prior_L=L0, 
                                                        model_L=L1)

    print(x_data.shape, θ_data.shape, post_mean.shape, post_var.shape)

    fig_x, fig_theta = plot_mvn_data(x_data, θ_data, feature_names=[f"dim{i}" for i in range(d)])
    plt.show()


    # 2) set up & train MDN
    key, tkey = random.split(key)

    model = build_vmapped_mdn(d = d, 
                              hidden= 2 * [8],
                              K = 4)

    t0 = time.perf_counter()
    state = train_multi_mdn(tkey, model,
                      x_data, θ_data,
                      lr=1e-4, 
                      n_epochs=500, 
                      batch_size=5000)
    t1 = time.perf_counter()

    print(f"Training took {(t1-t0):.2f}")

    # 3) save parameters
    ckpt_dir = os.path.abspath("./ckpt_multi_mdn")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir,
        target=state.params,
        step=0,
        prefix="multi_mdn_",
        overwrite=True
    )

    print("Saved MultiMDN params to", ckpt_dir)

    # #4) plot true CDF vs learned CDF for some x's
    # x_vals = jnp.array([[-5.0], [-4.0],[-2.0],[0.0],[2.0],[4.0], [5.0], [6.0]])  # shape [5,1]
    # θ_grid = jnp.linspace(-6, 6, 300)                     # evaluation points
    # plot_normal_cdf(model = model, params = state.params, x_vals=x_vals, θ_grid=θ_grid, p_mean=p_mean, p_var=p_var, d_var=d_var)

    # # x_vals = jnp.array([[.02],[.05],[1.],[2.0],[4.0]])  # shape [5,1]
    # # θ_grid = jnp.linspace(0, 4, 300)                     # evaluation points
    # # plot_gam_cdf(model=model, params=state.params, 
    # #              x_vals=x_vals, θ_grid=θ_grid, a = a, b = b,
    # #              m_x=m_x, m_θ=m_θ, std_x=std_x, std_θ=std_θ,
    # #              )

    # # 1) Create a grid of x-values
    # x_min, x_max = -10.0, 10.0
    # num_points   = 300
    # x_grid = jnp.linspace(x_min, x_max, num_points).reshape(-1,1)  # [num_points,1]

    # # 2) Run the MDN on the grid
    # logits, means, log_scales = model.apply(state.params, x_grid)  

    # # 3) Convert to numpy for plotting
    # x_np         = jnp.squeeze(x_grid).astype(float)
    # means_np     = jnp.array(means)
    # log_scales_np= jnp.array(log_scales)

    # # 4) Plot the K component-means vs x
    # plt.figure(figsize=(6,4))
    # for k in range(model.K):
    #     plt.plot(x_np, means_np[:,k], label=f"mean comp {k}")
    # plt.title("MDN Component Means vs x")
    # plt.xlabel("x")
    # plt.ylabel(r"$\mu_k(x)$")
    # plt.legend()
    # plt.grid(True)

    # # 5) Plot the log-scales vs x
    # plt.figure(figsize=(6,4))
    # for k in range(model.K):
    #     plt.plot(x_np, log_scales_np[:,k], label=f"log-scale comp {k}")
    # plt.title("MDN Component Log‑Scales vs x")
    # plt.xlabel("x")
    # plt.ylabel(r"$\log\sigma_k(x)$")
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()