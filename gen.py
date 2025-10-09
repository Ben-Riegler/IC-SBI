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

from ecdf import sig_marg_ecdf_vals

jax.config.update("jax_enable_x64", True)

class generator(nn.Module):
    emb_dim: int
    hidden_dims: List
    out_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndrray, x: jnp.ndrray) -> jnp.ndarray: 

        # z: (z_batch, z_dim)
        # x: (batch, x_dim)   
         
        z = z[None, :, : ] # (1, z_batch, z_dim)
        x = x[:, None, :] # (batch, 1, x_dim)

        h_z = nn.Dense(self.emb_dim)(z) # (1, z_batch, emb_dim)
        h_x = nn.Dense(self.emb_dim)(x) # (batch, 1, emb_dim)

        h = h_z + h_x # (batch, z_batch, emb_dim)

        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.swish(h)
        
        y = nn.Dense(self.out_dim)(h) # (batch, z_batch, out_dim)

        return y
 

@jax.jit
def train_step(state: train_state.TrainState,
               u: jnp.ndarray, # (batch, y_dim)
               x: jnp.ndarray, # (batch, x_dim)
               z: jnp.ndarray, # (z_batch, z_dim)
              ) -> Tuple[train_state.TrainState, jnp.ndarray]:
    
    K = z.shape[0]

    def loss_fn(params):
        y = state.apply_fn(params, z, x) # (batch, z_batch, y_dim)
        v = sig_marg_ecdf_vals(y) # (batch, z_batch, y_dim)

        u = u[:, None, :] # (batch, 1, y_dim)
        uv = jnp.linalg.vector_norm(u-v, axis=-1) # (batch, z_batch)

        dv = v[:, None, :, :] - v[:, :, None, :] # (batch, z_batch, z_batch, y_dim)
        vv = jnp.linalg.vector_norm(dv, axis=-1) # (batch, z_batch, z_batch)

        uv_m = jnp.mean(uv, axis=-1) # (batch)
        vv_m = jnp.mean(vv, axis=(-2,-1)) * K/(K-1) # (batch)

        diff = 2 * uv_m - vv_m # (batch)

        ed = jnp.mean(diff) # (1)

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
    
    Nx = u.shape[0]
    Nz = z.shape[0]

    losses = []

    state = create_train_state(key, model, learning_rate, z.shape, x.shape)

    for epoch in range(n_epochs):
        key, pkey1, pkey2 = random.split(key)
        x_idc = random.permutation(pkey1, range(Nx))
        z_idc = random.permutation(pkey2, range(Nz))

        x = x[x_idc]
        u = u[x_idc]
        z = z[z_idc]
        n_batches = Nx // batch_size

        for b in range(0, n_batches):

            x_batch = x[b*batch_size, (b+1)*batch_size]
            u_batch = u[b*batch_size, (b+1)*batch_size]
            z_batch = z[b*z_batch_size, (b+1)*z_batch_size]

            state, loss = train_step(state=state, u=u_batch, x=x_batch, z=z_batch)

            losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}  loss={loss:.4f}")


    return state, losses