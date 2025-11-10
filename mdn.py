import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
from typing import List, Any, Tuple
import time
from jax.scipy.stats import norm
from functools import partial
import itertools

jax.config.update("jax_enable_x64", True)

from data import gen_mv_normal_normal_data, mvn_posterior
from plots import plot_mvn_marginals, plot_losses
from utils import save_MDN, standardize

# -- model definition --------------------------------------------------------

class MDN(nn.Module):
    hidden_dims: List[int]
    K: int
    mean: jnp.ndarray | None = None
    std: jnp.ndarray | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray
                ) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        
        h = x if self.mean is None else standardize(self.mean, self.std)(x) 

        for dim in self.hidden_dims[:-1]:
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
        h = nn.Dense(self.hidden_dims[-1])(h)

        init_small_w = nn.initializers.normal(stddev=1e-3)
        init_zero_b = nn.initializers.constant(0.)

        logits    = nn.Dense(self.K)(h)   # (batch,K)
        means     = nn.Dense(self.K,
                             kernel_init=init_small_w, 
                             bias_init=init_zero_b
                             )(h)
        log_scales= nn.Dense(self.K, 
                             kernel_init=init_small_w, 
                             bias_init=init_zero_b
                             )(h)   

        return logits, means, log_scales


# -- loss, train step, state init -------------------------------------------

@jax.jit
def train_step(state: train_state.TrainState,
               x: jnp.ndarray,
               y: jnp.ndarray
              ) -> Tuple[train_state.TrainState, jnp.ndarray]:

    def loss_fn(params):
        logits, means, log_scales = state.apply_fn(params, x)
        log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True) # (batch, K)
        scales = jnp.exp(log_scales) # (batch, K)

        log_probs = (
            -0.5 * ((y - means)/scales)**2
            - log_scales
            - 0.5 * jnp.log(2*jnp.pi)
        ) # (batch, K)
        log_lik = jax.nn.logsumexp(log_pi + log_probs, axis=-1) # (batch, K)
        return -jnp.mean(log_lik)


    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss

@jax.jit
def val_loss_fun(state: train_state.TrainState,
                 x: jnp.ndarray,
                 y: jnp.ndarray
                 ):
        logits, means, log_scales = state.apply_fn(state.params, x)
        log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True) # (batch,K)
        scales = jnp.exp(log_scales) # (batch,K)

        log_probs = (
            -0.5 * ((y - means)/scales)**2
            - log_scales
            - 0.5 * jnp.log(2*jnp.pi)
        ) # (batch, K)
        log_lik = jax.nn.logsumexp(log_pi + log_probs, axis=-1) # (batch,K)
        return -jnp.mean(log_lik)
    



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

# -- training loop ----------------------------------------------------------

def train_mdn(keys: map, model, x_data, θ_data,
              lr, n_epochs, batch_size,
              x_val = None, theta_val = None, early_stop=100):
    

    n, x_dim = x_data.shape
    state = create_train_state(next(keys), model, lr, x_shape=(batch_size, x_dim))

    losses, val_losses = [], []


    for ep in range(1, n_epochs+1):
        perm = random.permutation(next(keys), n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = x_data[idx], θ_data[idx]

            if x_val is not None:
                val_loss = val_loss_fun(state, x_val, theta_val)
                val_losses.append(val_loss)

            state, loss = train_step(state, xb, yb)
            losses.append(loss)

        if len(val_losses) > early_stop and val_loss > max(val_losses[-early_stop:-1]):
            print("\nearly stop\n")
            break

        if ep % 10 == 0:
            if x_val is not None:
                print(f"Epoch {ep:03d}  train loss={loss:.4f}  val loss={val_loss:.4f}")
            else:
                print(f"Epoch {ep:03d}  train loss={loss:.4f}")

    return (state, losses) if x_val is None else (state, losses, val_losses)

def train_marginal_mdns(keys: map, model, x_data, θ_data,
              lr, n_epochs, batch_size, path,
              x_val = None, theta_val = None, early_stop=100):
    
    d = θ_data.shape[-1]
    
    par_list, losses_list, val_losses_list = [], [], []
    t0 = time.perf_counter()
    for dim in range(d):    
        θ_dat = θ_data[:, dim][:, None]
        train_results = train_mdn(keys, model,
                                        x_data, θ_dat,
                                        lr=lr, 
                                        n_epochs=n_epochs, 
                                        batch_size=batch_size,
                                        x_val = x_val, theta_val = theta_val[:, dim][:, None],
                                        early_stop=early_stop)
        
        par_list.append(train_results[0].params)
        losses_list.append(train_results[1])
        if x_val is not None:
            val_losses_list.append(train_results[2])

        save_MDN(path, dim, train_results[0].params)
            
    t1 = time.perf_counter()
    print(f"Saved all {d} MDNs, training took {(t1-t0):.2f}s")

    return (losses_list, par_list) if x_val is None else (losses_list, par_list, val_losses_list)


def get_cdf_vals(model, par_list, 
                 x_data, # (..., x_dim)
                 θ_data, # (..., θ_dim)
                 ):

    """Get marginal CDF values F(θ_j|x) from trained MDN"""
    d = θ_data.shape[-1]
    u = []
    # Loop over dimensions
    for dim in range(d):

        logits, means, log_scales = model.apply(par_list[dim], x_data)  # each: (..., K)
        scales = jnp.exp(log_scales) 
        log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)  # (..., K)
        pi = jnp.exp(log_pi)                                           # (..., K)

        # Learned mixture CDF
        comp_cdfs = norm.cdf((θ_data[..., dim][..., None] - means) / scales)  # (..., K)
        u_ = jnp.sum(pi * comp_cdfs, axis=-1) # (...)
        u.append(u_)

    u = jnp.stack(u, axis=-1) # (..., θ_dim)

    return u


# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    # set up
    path = "mdn/"
    os.makedirs(path, exist_ok=True)

    d = 5
    K = 1   
    N = 1000
    N_val = 5000
    batch_size = 1000
    epochs = 5000
    early_stop = 100

    root_key = random.key(42)
    keys = map(partial(random.fold_in, root_key), itertools.count())
    L0 = random.normal(next(keys), (d,d))
    L1 = random.normal(next(keys), (d,d))

    # L0 = jnp.sqrt(0.1) * jnp.eye(d)
    # L1 = jnp.sqrt(0.1) * jnp.eye(d)

    prior_mean = jnp.zeros((d,1))

    x_data, θ_data = gen_mv_normal_normal_data(next(keys), 
                                                n_samples=N, 
                                                prior_mean=prior_mean,
                                                prior_L=L0, 
                                                model_L=L1)
    
    x_val, theta_val = gen_mv_normal_normal_data(next(keys), 
                                                n_samples=N_val, 
                                                prior_mean=prior_mean,
                                                prior_L=L0, 
                                                model_L=L1)
    
    post_mean, post_var = mvn_posterior(x_data, prior_mean, L0, L1)

    print(x_data.shape, θ_data.shape, post_mean.shape, post_var.shape)

    model = MDN(hidden_dims= 2 * [32], 
                K=K, 
                mean=x_data.mean(axis=0), 
                std=x_data.std(axis=0))

    losses_list, par_list, val_losses_list = train_marginal_mdns(keys, model, x_data, θ_data, 1e-4, epochs, batch_size, "mdn/", 
                                                x_val=x_val, theta_val=theta_val, early_stop=early_stop)

    plot_losses(losses_list, path, val_losses_list)

    test_ids = random.choice(next(keys), N, (4,))
    x_test = x_val[test_ids]
    plot_mvn_marginals(model, par_list, x_test, prior_mean, L0, L1, theta_range=(-5, 5), path = path)





