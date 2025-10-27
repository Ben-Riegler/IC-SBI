import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from typing import List, Any, Tuple
import time

from ecdf import sig_marg_ecdf_vals
from plots import plot_loss
from utils import save_gen

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

class generator(nn.Module):
    emb_dim: int
    hidden_dims: List
    out_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray: 

        # batch dims need to be same (see `train_generator`)
        # z: (..., z_dim)
        # x: (..., x_dim)   

        h_z = nn.Dense(self.emb_dim)(z) # (..., emb_dim)
        h_x = nn.Dense(self.emb_dim)(x) # (..., emb_dim)
        h = jnp.concatenate([h_z, h_x], axis=-1) # (..., 2*emb_dim)

        # h = h_z + h_x # (..., emb_dim)

        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.swish(h)
        
        y = nn.Dense(self.out_dim)(h) # (..., out_dim)

        return y
 

@jax.jit
def train_step(state: train_state.TrainState,
               u: jnp.ndarray, # (batch, y_dim)
               x: jnp.ndarray, # (batch, x_dim)
               z: jnp.ndarray, # (z_batch, z_dim)
              ) -> Tuple[train_state.TrainState, jnp.ndarray]:
    
    K = z.shape[0]
    i_idx, j_idx = jnp.triu_indices(K, k=1) # upper triangular indices 

    def loss_fn(params):

        # expand, we want all samples z to interact with each sample in x
        z_ = z[None, :, : ] # (1, z_batch, z_dim)
        x_ = x[:, None, :] # (batch, 1, x_dim)
        tmp = z_ + x_
        tmp = jnp.zeros_like(tmp)
        z_ += tmp # (batch, z_batch, y_dim)
        x_ += tmp # (batch, z_batch, y_dim)
        y = state.apply_fn(params, z_, x_) # (batch, z_batch, y_dim)

        v = sig_marg_ecdf_vals(y) # (batch, z_batch, y_dim)

        u_ = u[:, None, :] # (batch, 1, y_dim)
        uv = jnp.linalg.vector_norm(u_-v, axis=-1) # (batch, z_batch)

        dv = v[:, i_idx, :] - v[:, j_idx, :] # (batch, (K^2-K)/2, y_dim)
        vv = jnp.linalg.vector_norm(dv, axis=-1) # (batch, (K^2-K)/2)
  
        uv_m = jnp.mean(uv, axis=-1) # (batch)
        vv_m = jnp.mean(vv, axis=-1) # (batch)

        diff = 2 * uv_m - vv_m # (batch)

        ed = jnp.mean(diff) # ()

        return ed


    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss

def create_train_state(rng: Any, model: generator,
                       learning_rate: float,
                       z_shape: Tuple[int,int],
                       x_shape: Tuple[int,int]
                      ) -> train_state.TrainState:
    
    """Initial training state, will be updated in `train_step`"""

    params = model.init(rng, jnp.zeros(z_shape), jnp.zeros(x_shape)) # model is stateless, not affected by calling init
    tx     = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def train_generator(key,
                    model:generator,
                    u: jnp.ndarray,
                    x: jnp.ndarray,
                    z: jnp.ndarray,
                    learning_rate: float,
                    n_epochs: int,
                    batch_size: int,
                    z_batch_size: int,
                    ):
    
    Nx, x_dim = x.shape
    Nz, z_dim = z.shape


    losses = []

    state = create_train_state(key, model, learning_rate, 
                               (z_dim),
                               (x_dim)
                               )
    n_batches = Nx // batch_size


    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        key, pkey1, pkey2 = random.split(key, 3)
        x_idc = random.permutation(pkey1, Nx)
        z_idc = random.permutation(pkey2, Nz)

        x = x[x_idc]
        u = u[x_idc]
        z = z[z_idc]
        
        for b in range(0, n_batches):

            x_batch = x[b*batch_size : (b+1)*batch_size]
            u_batch = u[b*batch_size : (b+1)*batch_size]
            z_batch = z[b*z_batch_size : (b+1)*z_batch_size]

            state, loss = train_step(state=state, u=u_batch, x=x_batch, z=z_batch)

            losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}  loss={loss:.4f}")
        
    t1 = time.perf_counter()

    print(f"Training took {(t1-t0):.2f}s")


    return state, losses


if __name__ == "__main__":

    path = "experiments/gen/"

    N = 5000
    d = 4
    n_epochs = 1000

    batch_size = 5000
    z_batch_size = 20
    Nz = z_batch_size * N // batch_size

    key = random.PRNGKey(1)

    key, key1, key2, key3, key4 = random.split(key, 5)
    u = random.uniform(key2, (N, d))
    x = random.normal(key1, (N, d))
    z = random.normal(key3, (Nz, d))

    model = generator(emb_dim=8, 
                      hidden_dims= 3 * [8], 
                      out_dim=d)

    state, losses = train_generator(key4, 
                                    model, 
                                    u, x, z, 
                                    1e-4, n_epochs, batch_size, z_batch_size)
    save_gen(path, state.params)
    
    plot_loss(losses, path)