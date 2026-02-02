import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import matplotlib.pyplot as plt
from typing import List, Any, Tuple
import math
import matplotlib.pyplot as plt
import time
from data import gen_mv_normal_normal_data
from plots import mvn_posterior, plot_multi_mvn_marginals, plot_mvn_data, plot_loss
from mdn import MDN
from utils import save_multiMDN

jax.config.update("jax_enable_x64", True)

# def build_vmapped_mdn(d: int, hidden: List[int], K: int):
#     """
#     Returns a module that replicates MDN d times with independent parameters.
#     Output shapes: (batch, d, K) for each head’s output.
#     """
#     # Vectorize MDN over a new axis of size d, duplicating params per axis element
#     VMapped = nn.vmap(
#         MDN,
#         variable_axes={"params": 0},   # separate params per copy
#         split_rngs={"params": True},   # different init per copy
#         in_axes=None,                  # same x goes to every copy
#         out_axes=1,                    # stack outputs dim axis (1)
#         axis_size=d
#     )

#     class MultiMDN(nn.Module):
#         hidden_dims: List[int]
#         K: int

#         @nn.compact
#         def __call__(self, x: jnp.ndarray):
#             # x (batch, d)
#             logits, means, log_scales = VMapped(self.hidden_dims, self.K)(x) # (batch, d, K)
#             return logits, means, log_scales

#     return MultiMDN(hidden, K)


class SharedEmbedMultiMDN(nn.Module):
    var_dim: int
    hidden_dims: List[int]
    K: int

    def setup(self):
        # vmaped dimension heads
        self.BatchDense = nn.vmap(nn.Dense,
                            in_axes=None, # all share input, no axis assignment here
                            out_axes=1,
                            axis_size=self.var_dim,
                            variable_axes={"params": 0},
                            split_rngs={"params": True}
                            )

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray # (B, x_dim)
                ) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        h = x 

        # feature extactor shared across variable dims
        for dim in self.hidden_dims[:-1]:
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
        h = nn.Dense(self.hidden_dims[-1])(h)

        init_small_w = nn.initializers.normal(stddev=1e-3)
        init_zero_b = nn.initializers.constant(0.)

        # independent head for each variable dim

        logits = self.BatchDense(self.K)(h) # (B, var_dim, K)
        means = self.BatchDense(self.K,
                            kernel_init=init_small_w, 
                            bias_init=init_zero_b
                            )(h)
        log_scales = self.BatchDense(self.K, 
                                kernel_init=init_small_w, 
                                bias_init=init_zero_b
                                )(h) 

        # const variance inductive bias
        # const = jnp.ones((h.shape[0], 1), dtype=h.dtype)   # (B, 1)
        # log_scales = self.BatchDense(self.K,
        #                             kernel_init=init_small_w,
        #                             bias_init=init_zero_b
        #                             )(const)               # (B, var_dim, K)


        return logits, means, log_scales

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

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss

def create_train_state(rng: Any, model,
                       learning_rate: float,
                       batch_size: Tuple[int,int],
                       d: int
                      ) -> train_state.TrainState:
    
    """Initial training state, will be updated in `train_step`"""

    params = model.init(rng, jnp.zeros((batch_size, d))) # model is stateless, not affected by calling init
    tx     = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def train_multi_mdn(rng, model, x_data, θ_data,
              lr, n_epochs, batch_size):
    n, d = x_data.shape
    state = create_train_state(rng, model, lr, batch_size, d)
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
    
    return state, losses

# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    # set up
    path = "multi_mdn/"
    os.makedirs(path, exist_ok=True)

    d = 10
    K = 1
    N = 1000
    batch_size = 5000
    epochs = 10000

    key = random.PRNGKey(1)
    key, k1, k2, k3 = random.split(key, 4)
    L0 = random.normal(k1, (d,d))
    L1 = random.normal(k2, (d,d))

    prior_mean = jnp.zeros((d,1))

    x_data, θ_data = gen_mv_normal_normal_data(key, 
                                                        n_samples=N, 
                                                        prior_mean=prior_mean,
                                                        prior_L=L0, 
                                                        model_L=L1)
    
    post_mean, post_var = mvn_posterior(x_data, prior_mean, L0, L1)

    print(x_data.shape, θ_data.shape, post_mean.shape, post_var.shape)

    fig_x, fig_theta = plot_mvn_data(x_data, θ_data, feature_names=[f"dim{i}" for i in range(d)])
    plt.show()

    key, tkey = random.split(key)

    # model = build_vmapped_mdn(d = d, 
    #                           hidden= 2 * [8],
    #                           K = K)

    model = SharedEmbedMultiMDN(var_dim=d, hidden_dims = 1*[16], K = K)

    t0 = time.perf_counter()
    state, losses = train_multi_mdn(tkey, model,
                                        x_data, θ_data,
                                        lr=1e-4, 
                                        n_epochs=epochs, 
                                        batch_size=batch_size)
    t1 = time.perf_counter()

    print(f"Training took {(t1-t0):.2f}")

    save_multiMDN(path=path, params=state.params)


    plot_loss(losses, path)

    key, tk = random.split(key)

    test_ids = random.choice(tk, N, (4,))
    x_test = x_data[test_ids]
    fig, axes = plot_multi_mvn_marginals(model, state.params, x_test, prior_mean, L0, L1, theta_range=(-5, 5))
    plt.savefig(path + "fit.pdf")
    plt.close()