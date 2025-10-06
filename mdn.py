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
import time

# -- model definition --------------------------------------------------------

class MDN(nn.Module):
    hidden_dims: List[int]
    K: int

    @nn.compact
    def __call__(self, x: jnp.ndarray
                ) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        h = x
        for dim in self.hidden_dims[:-1]:
            h = nn.Dense(dim)(h)
            h = nn.swish(h)
        h = nn.Dense(self.hidden_dims[-1])(h)

        logits    = nn.Dense(self.K)(h)   # [batch,K]
        means     = nn.Dense(self.K)(h)
        log_scales= nn.Dense(self.K)(h)   

        return logits, means, log_scales
    

def load_model(ckpt_dir, model_def, rng_key, hidden_dims= 1 * [8], K = 1, d = 1):
    """
    Load a trained MDN model from checkpoint in a notebook.

    Args:
      ckpt_dir: directory containing checkpoints
      model_def: MDN class definition
      rng_key: PRNGKey for parameter init
      hidden_dims: same architecture used during training

    Returns:
      model: Flax module instance
      params: loaded parameters
    """
    # Initialize model and dummy params
    model = model_def(hidden_dims= hidden_dims, K = K)
    dummy_x = jnp.zeros((d,1))
    init_vars = model.init(rng_key, dummy_x)

    # Restore
    restored = checkpoints.restore_checkpoint(
        ckpt_dir,
        target={'params': init_vars['params']},
        prefix='mdn_'
    )
    return model, restored
    

# -- loss, train step, state init -------------------------------------------

@jax.jit
def train_step(state: train_state.TrainState,
               x: jnp.ndarray,
               y: jnp.ndarray
              ) -> Tuple[train_state.TrainState, jnp.ndarray]:

    def loss_fn(params):
        logits, means, log_scales = state.apply_fn(params, x)
        log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True) # [B,K]
        scales = jnp.exp(log_scales) # [B,K]

        log_probs = (
            -0.5 * ((y - means)/scales)**2
            - log_scales
            - 0.5 * jnp.log(2*jnp.pi)
        ) # [B,K]
        log_lik = jax.nn.logsumexp(log_pi + log_probs, axis=-1) # [B]
        return -jnp.mean(log_lik)

    grads = jax.grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    loss  = loss_fn(state.params)
    return state, loss

def create_train_state(rng: Any, model: MDN,
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

# -- synthetic Normal‐Normal data -------------------------------------------

def generate_normal_normal_data(key,
                                n_samples=10000,
                                prior_var=1.0,
                                data_var=1.0,
                                prior_mean=0.0):
    k1,k2 = random.split(key)
    θ = prior_mean + random.normal(k1, (n_samples,)) * jnp.sqrt(prior_var)
    x = θ + random.normal(k2,(n_samples,)) * jnp.sqrt(data_var)

    # true posterior parameters:
    post_var  = 1.0/(1.0/prior_var + 1.0/data_var)
    post_mean = (prior_mean*data_var + x*prior_var) / (data_var + prior_var)

    return (x.reshape(-1,1),
            θ.reshape(-1,1),
            post_mean, post_var)

def generate_exp_gam_data(key,
                          a = 1,
                          b = 1,
                          n_samples = 10000):
    
    k1,k2 = random.split(key)

    θ = random.gamma(k1, a, shape=(n_samples, 1)) / b
    x = random.exponential(k2, shape=(n_samples, 1)) / θ

    return x, θ

# -- plot ----------------------------------------------------------
def plot_normal_cdf(model, params, x_vals, θ_grid, p_mean, p_var, d_var):

    post_var = 1. / (1./ p_var + 1./ d_var)
    post_std = jnp.sqrt(post_var)
    cols = 3
    n_plts = len(x_vals)
    rows = n_plts // cols + 1
    fig, ax = plt.subplots(rows, cols, sharey=True)
    axes = ax.flatten()
    for idx, x0 in enumerate(x_vals):
        ax = axes[idx]
        # true CDF: Φ((θ - post_mean(x0)) / post_std)
        μ0 = (p_mean*d_var + x0*p_var) / (d_var + p_var)  
        true_cdf = norm.cdf((θ_grid - μ0) / post_std)

        # learned CDF:
        #  - broadcast x0 to [300,1]
        x_b = jnp.tile(x0, (θ_grid.shape[0],1))
        logits, means, log_scales = model.apply(params, x_b)
        π = jax.nn.softmax(logits, axis=-1)      # [300,K]
        σ = jnp.exp(log_scales)                  # [300,K]
        # component‐wise CDFs
        comp_cdfs = norm.cdf((θ_grid[:,None] - means)/σ)  # [300,K]
        learned_cdf = jnp.sum(π * comp_cdfs, axis=-1)     # [300]

        
        ax.plot(θ_grid, learned_cdf,    label="MDN CDF")
        ax.plot(θ_grid, true_cdf, '--', label="True CDF")
        ax.set_title(f"CDF @ x = {float(x0.item()):.1f}")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("CDF")
        ax.legend()

    for ax in range(n_plts, rows*cols):
        axes[ax].remove()

    fig.tight_layout()
    fig.show()

def plot_gam_cdf(model, params, x_vals, θ_grid, a, b, m_x, m_θ, std_x, std_θ):

    for x0 in x_vals:
        x = x0
        true_cdf = gamma.cdf(θ_grid, a = a+1, scale= 1/(b+x0) )

        # learned CDF:
        #  - broadcast x0 to [300,1]
        x0 = jnp.log(x0)
        x0 = (x0 - m_x) / std_x

        x_b = jnp.tile(x0, (θ_grid.shape[0],1))
        logits, means, log_scales = model.apply(params, x_b)
        π = jax.nn.softmax(logits, axis=-1)      # [300,K]
        σ = jnp.exp(log_scales)                  # [300,K]
        # component‐wise CDFs
        z_grid = jnp.log(θ_grid)
        z_grid = (z_grid - m_θ) / std_θ

        comp_cdfs = norm.cdf((z_grid[:,None] - means)/σ)  # [300,K]
        learned_cdf = jnp.sum(π * comp_cdfs, axis=-1)     # [300]

        plt.figure()
        plt.plot(θ_grid, learned_cdf,    label="MDN CDF")
        plt.plot(θ_grid, true_cdf, '--', label="True CDF")
        plt.title(f"CDF @ x = {float(x.item()):.2f}")
        plt.xlabel(r"$\theta$")
        plt.ylabel("CDF")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_losses(losses):

    fig, ax = plt.subplots(1,1)

    n = len(losses)
    ns = [i for i in range(n)]

    ax.plot(ns, losses, color="red", linewidth=0.5)
    ax.set_title("Training loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")

    fig.show()


# -- training loop ----------------------------------------------------------

def train_mdn(rng, model, x_data, θ_data,
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
    key = random.PRNGKey(0)
    key, d_key = random.split(key)

    p_var, d_var, p_mean = 1., 1., 0.0
    x_data, θ_data, _, _ = \
        generate_normal_normal_data(d_key,
                                    prior_var=p_var,
                                    data_var=d_var,
                                    prior_mean=p_mean,
                                    n_samples = 5000)
    # a, b = 4., 2.
    # x_data, θ_data = generate_exp_gam_data(d_key,
    #                                        a = a,
    #                                        b = b,
    #                                        n_samples=5000)

    print(f"x range: ({x_data.min()}, {x_data.max()})")
    print(f"θ range: ({θ_data.min()}, {θ_data.max()})")

    # x_data, θ_data = jnp.log(x_data), jnp.log(θ_data)

    # m_x, std_x = jnp.mean(x_data), jnp.std(x_data)
    # m_θ, std_θ = jnp.mean(θ_data), jnp.std(θ_data) 

    # x = (x_data - m_x) / std_x
    # θ = (θ_data - m_θ) /  std_θ

    # print(f"x range: ({x.min()}, {x.max()})")
    # print(f"θ range: ({θ.min()}, {θ.max()})")

    # 2) set up & train MDN
    key, tkey = random.split(key)

    model = MDN(hidden_dims= 2 * [8], 
                K=1)
    

    t0 = time.perf_counter()
    state = train_mdn(tkey, model,
                      x_data, θ_data,
                      lr=1e-4, 
                      n_epochs=10000, 
                      batch_size=5000)
    t1 = time.perf_counter()

    print(f"Training took {(t1-t0):.2f}s")

    # 3) save parameters
    ckpt_dir = os.path.abspath("./ckpt_mdn")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir,
        target=state.params,
        step=0,
        prefix="mdn_",
        overwrite=True
    )
    print("Saved MDN params to", ckpt_dir)

    #4) plot true CDF vs learned CDF for some x's
    x_vals = jnp.array([[-5.0], [-4.0],[-2.0],[0.0],[2.0],[4.0], [5.0], [6.0]])  # shape [5,1]
    θ_grid = jnp.linspace(-6, 6, 300)                     # evaluation points
    plot_normal_cdf(model = model, params = state.params, x_vals=x_vals, θ_grid=θ_grid, p_mean=p_mean, p_var=p_var, d_var=d_var)

    # x_vals = jnp.array([[.02],[.05],[1.],[2.0],[4.0]])  # shape [5,1]
    # θ_grid = jnp.linspace(0, 4, 300)                     # evaluation points
    # plot_gam_cdf(model=model, params=state.params, 
    #              x_vals=x_vals, θ_grid=θ_grid, a = a, b = b,
    #              m_x=m_x, m_θ=m_θ, std_x=std_x, std_θ=std_θ,
    #              )

    # 1) Create a grid of x-values
    x_min, x_max = -10.0, 10.0
    num_points   = 300
    x_grid = jnp.linspace(x_min, x_max, num_points).reshape(-1,1)  # [num_points,1]

    # 2) Run the MDN on the grid
    logits, means, log_scales = model.apply(state.params, x_grid)  

    # 3) Convert to numpy for plotting
    x_np         = jnp.squeeze(x_grid).astype(float)
    means_np     = jnp.array(means)
    log_scales_np= jnp.array(log_scales)

    # 4) Plot the K component-means vs x
    plt.figure(figsize=(6,4))
    for k in range(model.K):
        plt.plot(x_np, means_np[:,k], label=f"mean comp {k}")
    plt.title("MDN Component Means vs x")
    plt.xlabel("x")
    plt.ylabel(r"$\mu_k(x)$")
    plt.legend()
    plt.grid(True)

    # 5) Plot the log-scales vs x
    plt.figure(figsize=(6,4))
    for k in range(model.K):
        plt.plot(x_np, log_scales_np[:,k], label=f"log-scale comp {k}")
    plt.title("MDN Component Log‑Scales vs x")
    plt.xlabel("x")
    plt.ylabel(r"$\log\sigma_k(x)$")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

