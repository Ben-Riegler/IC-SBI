import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from typing import List, Any, Tuple
import time
from functools import partial
import itertools

from ecdf import sig_marg_ecdf_vals
from plots import plot_loss
from utils import save_gen, standardize

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_debug_nans", False)

class generator(nn.Module):
    emb_dim: int
    hidden_dims: List
    out_dim: int
    x_mean: jnp.ndarray | None = None
    x_std: jnp.ndarray | None = None

    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray: 

        # batch dims need to be same (see `train_generator`)
        # z: (..., z_dim)
        # x: (..., x_dim)  

        x = x if self.x_mean is None else standardize(self.x_mean, self.x_std)(x)

        h_z = nn.Dense(self.emb_dim)(z) # (..., emb_dim)
        h_x = nn.Dense(self.emb_dim)(x) # (..., emb_dim)
        h = jnp.concatenate([h_z, h_x], axis=-1) # (..., 2*emb_dim)

        # h = h_z + h_x # (..., emb_dim)

        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.swish(h)
        
        y = nn.Dense(self.out_dim)(h) # (..., out_dim)

        return y
 
# jit and free up buffer of state since it will not be used anymore after this call 
@partial(jax.jit, donate_argnames="state")
def train_step(state: train_state.TrainState,
               u: jnp.ndarray, # (batch, y_dim)
               x: jnp.ndarray, # (batch, x_dim)
               z: jnp.ndarray, # (K, z_dim)
              ) -> Tuple[train_state.TrainState, jnp.ndarray]:
    
    B, x_dim = x.shape
    K, z_dim = z.shape
    _K = min(200, K)
    i_idx, j_idx = jnp.triu_indices(_K, k=1) # upper triangular indices 

    def loss_fn(params):

        # expand, we want all samples z to interact with each sample in x
        
        z_ = jnp.broadcast_to(z[None, :, :], (B, K, z_dim)); # print("z_", z_.shape)
        x_ = jnp.broadcast_to(x[:, None, :], (B, K, x_dim)); # print("x_", x_.shape)
        y = state.apply_fn(params, z_, x_); # print("y", y.shape) # (batch, K, y_dim)

        v = sig_marg_ecdf_vals(y)[:, :_K, :]; # print("v", v.shape) # (batch, _K, y_dim)

        u_ = u[:, None, :]; # print("u_", u_.shape) # (batch, 1, y_dim)
        uv = jnp.linalg.vector_norm(u_-v, axis=-1); # print("uv", uv.shape) # (batch, _K)

        dv = v[:, i_idx, :] - v[:, j_idx, :]; # print("dv", dv.shape) # (batch, (_K^2-_K)/2, y_dim)
        vv = jnp.linalg.vector_norm(dv, axis=-1); # print("vv", vv.shape) # (batch, (_K^2-_K)/2 )
  
        uv_m = jnp.mean(uv, axis=-1); # print("uv_m", uv_m.shape) # (batch)
        vv_m = jnp.mean(vv, axis=-1); # print("vv_m", vv_m.shape) # (batch)

        diff = 2 * uv_m - vv_m; # print("diff", diff.shape) # (batch)

        ed = jnp.mean(diff); # print("ed", ed.shape) # ()

        return ed

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss

@jax.jit
def val_loss_fn(state: train_state.TrainState,
                u: jnp.ndarray, # (batch, y_dim)
                x: jnp.ndarray, # (batch, x_dim)
                z: jnp.ndarray, # (z_batch, z_dim)
                ):
    
    B, x_dim = x.shape
    K, z_dim = z.shape
    _K = min(10, K)
    i_idx, j_idx = jnp.triu_indices(_K, k=1) # upper triangular indices 

    # expand, we want all samples z to interact with each sample in x
    z_ = jnp.broadcast_to(z[None, :, :], (B, K, z_dim))
    x_ = jnp.broadcast_to(x[:, None, :], (B, K, x_dim))
    y = state.apply_fn(state.params, z_, x_) # (batch, z_batch, y_dim)

    v = sig_marg_ecdf_vals(y)[:, :_K, :] # (batch, _K, y_dim) # (batch, z_batch, y_dim)

    u_ = u[:, None, :] # (batch, 1, y_dim)
    uv = jnp.linalg.vector_norm(u_-v, axis=-1) # (batch, z_batch)

    dv = v[:, i_idx, :] - v[:, j_idx, :] # (batch, (_K^2-_K)/2, y_dim)
    vv = jnp.linalg.vector_norm(dv, axis=-1) # (batch, (_K^2-_K)/2)

    uv_m = jnp.mean(uv, axis=-1) # (batch)
    vv_m = jnp.mean(vv, axis=-1) # (batch)

    diff = 2 * uv_m - vv_m # (batch)

    ed = jnp.mean(diff) # ()

    return ed
    


def create_train_state(key: Any, model: generator,
                       learning_rate: float,
                       z_shape: Tuple[int,int],
                       x_shape: Tuple[int,int]
                      ) -> train_state.TrainState:
    
    """Initial training state, will be updated in `train_step`"""

    params = model.init(key, jnp.zeros(z_shape), jnp.zeros(x_shape)) # model is stateless, not affected by calling init
    tx     = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def train_generator(key: map,
                    model: generator,
                    u: jnp.ndarray,
                    x: jnp.ndarray,
                    learning_rate: float,
                    n_epochs: int,
                    batch_size: int,
                    z_batch_size: int,
                    z_dim: int,
                    u_val: jnp.ndarray | None = None,
                    x_val: jnp.ndarray | None = None,
                    early_stop: int | None = None
                    ):
    
    N, x_dim = x.shape

    z_val = random.normal(next(key), (z_batch_size, z_dim))

    losses, val_losses = [], []

    state = create_train_state(next(key), model, learning_rate, 
                               (z_dim),
                               (x_dim)
                               )
    n_batches = N // batch_size
    assert n_batches > 0

    # z_batch = random.normal(random.key(1), (z_batch_size, z_dim)) # same each batch 

    t0 = time.perf_counter()
    for epoch in range(n_epochs):

        x_idc = random.permutation(next(key), N)
        x = x[x_idc]
        u = u[x_idc]
       
        for b in range(0, n_batches):

            x_batch = x[b*batch_size : (b+1)*batch_size]
            u_batch = u[b*batch_size : (b+1)*batch_size]
            # z_batch = random.normal(next(key), (z_batch_size, z_dim)) # resample each batch
            z_batch = random.normal(random.key(b), (z_batch_size, z_dim)) # fix z for given batch across epochs
            
            if x_val is not None:
                val_loss = val_loss_fn(state=state, u = u_val, x = x_val, z = z_val)
                val_losses.append(val_loss)

            state, loss = train_step(state=state, u=u_batch, x=x_batch, z=z_batch)
            losses.append(loss)
        
        if len(val_losses) > early_stop and val_loss > max(val_losses[-early_stop:-1]):
            print("\nearly stop\n")
            break

        if epoch % 10 == 0:
            if x_val is not None:
                print(f"Epoch {epoch:03d}  train loss={loss:.4f}  val loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch:03d}  train loss={loss:.4f}")
        
    t1 = time.perf_counter()

    print(f"Training took {(t1-t0):.2f}s")

    return (state, losses) if x_val is None else (state, losses, val_losses)



if __name__ == "__main__":

    root_key = random.key(3)
    keys = map(partial(random.fold_in, root_key), itertools.count())

    path = "gen/"

    N = 1000
    N_val = 10000
    d = 10
    n_epochs = 400
    early_stop = 1000000

    batch_size = 1000
    z_batch_size = 800
    
    u = random.uniform(next(keys), (N, d))
    x = random.normal(next(keys), (N, d))

    u_val = random.uniform(next(keys), (N_val, d))
    x_val = random.normal(next(keys), (N_val, d))

    model = generator(emb_dim=32, 
                      hidden_dims=[128], 
                      out_dim=d,
                      x_mean=x.mean(axis=0), 
                      x_std=x.std(axis=0)
                      )

    state, losses, val_losses = train_generator(keys, 
                                                model, 
                                                u, x,  
                                                1e-3, n_epochs, batch_size, z_batch_size, d,
                                                u_val=u_val, x_val=x_val,
                                                early_stop=early_stop
                                                )
    save_gen(path, state.params)
    
    plot_loss(losses, path, val_losses)
