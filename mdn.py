import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import matplotlib.pyplot as plt
from typing import List, Any, Tuple
import time

jax.config.update("jax_enable_x64", True)

from data import gen_mv_normal_normal_data, mvn_posterior
from plots import plot_mvn_marginals, plot_losses

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

        logits    = nn.Dense(self.K)(h)   # (batch,K)
        means     = nn.Dense(self.K)(h)
        log_scales= nn.Dense(self.K)(h)   

        return logits, means, log_scales


# -- loss, train step, state init -------------------------------------------

@jax.jit
def train_step(state: train_state.TrainState,
               x: jnp.ndarray,
               y: jnp.ndarray
              ) -> Tuple[train_state.TrainState, jnp.ndarray]:

    def loss_fn(params):
        logits, means, log_scales = state.apply_fn(params, x)
        log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True) # (batch,K)
        scales = jnp.exp(log_scales) # (batch,K)

        log_probs = (
            -0.5 * ((y - means)/scales)**2
            - log_scales
            - 0.5 * jnp.log(2*jnp.pi)
        ) # (batch,K)
        log_lik = jax.nn.logsumexp(log_pi + log_probs, axis=-1) # (batch,K)
        return -jnp.mean(log_lik)


    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

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

    return state, losses

def train_marginal_mdns(key, model, x_data, θ_data,
              lr, n_epochs, batch_size, path):
    
    key, tkey = random.split(key)
    par_list = []
    losses_list = []
    t0 = time.perf_counter()
    for dim in range(d):

        
        
        θ_dat = θ_data[:, dim][:, None]
        state, losses = train_mdn(tkey, model,
                                        x_data, θ_dat,
                                        lr=lr, 
                                        n_epochs=n_epochs, 
                                        batch_size=batch_size)
        
        par_list.append(state.params)
        losses_list.append(losses)
        
        ckpt_dir = os.path.join(path, f"ckpt_mdn_{dim}")
        ckpt_dir = os.path.abspath(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoints.save_checkpoint(
            ckpt_dir,
            target=state.params,
            step=0,
            prefix="mdn_",
            overwrite=True
        )
        print("Saved MDN params to", ckpt_dir)
        
    t1 = time.perf_counter()
    print(f"Training took {(t1-t0):.2f}s")

    return losses_list, par_list

# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    # set up
    path = "mdn/"
    os.makedirs(path, exist_ok=True)

    d = 4
    K = 4
    N = 5000
    batch_size = 5000
    epochs = 5000

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

    key, trkey = random.split(key)

    model = MDN(hidden_dims= 2 * [8], 
                K=K)

    losses_list, par_list = train_marginal_mdns(trkey, model, x_data, θ_data, 1e-4, epochs, batch_size, "mdn/")

    plot_losses(losses_list, path)


    key, tk = random.split(key)
    test_ids = random.choice(tk, N, (4,))
    x_test = x_data[test_ids]
    fig, axes = plot_mvn_marginals(model, par_list, x_test, prior_mean, L0, L1, theta_range=(-5, 5))
    plt.savefig(path + "fit.pdf")
    plt.close()




