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
from jax.nn import softmax
jax.config.update("jax_enable_x64", True)

from data import gen_mv_normal_normal_data, mvn_posterior
from plots import plot_mvn_marginals, plot_loss
from utils import save_gen, standardize

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# -- model definition --------------------------------------------------------

class GMM(nn.Module):
    hidden_dims: List[int]
    K: int
    d: int
    # mean: jnp.ndarray | None = None
    # std: jnp.ndarray | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray
                ) -> Tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        
        batch = x.shape[0]
        l_lmnts = (self.d**2 + self.d) // 2
        
        # h = x if self.mean is None else standardize(self.mean, self.std)(x) 
        h = x
        for dim in self.hidden_dims[:-1]:
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
        h = nn.Dense(self.hidden_dims[-1])(h)

        # init_small_w = nn.initializers.normal(stddev=1e-3)
        # init_zero_b = nn.initializers.constant(0.)

        logits    = nn.Dense(self.K)(h)   # (batch,K)
        means     = nn.Dense(self.K*self.d,
                            #  kernel_init=init_small_w, 
                            #  bias_init=init_zero_b
                             )(h)
        chol_pars = nn.Dense(self.K*l_lmnts, 
                            #  kernel_init=init_small_w, 
                            #  bias_init=init_zero_b
                             )(h)   
        
        means = means.reshape(batch, self.K, self.d)
        chol_pars = chol_pars.reshape(batch, self.K, l_lmnts)

        return logits, means, chol_pars
    
def chol_pars_to_L(chol_pars, # (batch, K, l_lmnts)
                   d                 
                   ):
    
    batch, K, l_lmnts = chol_pars.shape

    # d = int((jnp.sqrt(1 + 8 * l_lmnts) - 1) / 2)

    Ls = jnp.zeros((batch, K, d, d))

    idx = jnp.tril_indices(d)

    # fill lower triangle
    Ls = Ls.at[..., idx[0], idx[1]].set(chol_pars)

    # enforce positive diagonal
    diag = jnp.diagonal(Ls, axis1=-2, axis2=-1)
    diag = nn.softplus(diag) + 1e-5
    Ls = Ls.at[..., jnp.arange(d), jnp.arange(d)].set(diag)

    return Ls # (batch, K, d, d)

def sample_GMM_gen(key, N, 
                   logits, # (B, K)
                   means, # (B, K, d)
                   chol_pars # (batch, K, l_lmnts)
                   ):

    # for each x in batch, sample N times: y (B, N, d)

    key, k_key, e_key = random.split(key, 3)
    B, K, d = means.shape
    Ls = chol_pars_to_L(chol_pars, d) # (B, K, d, d)

    ks = random.categorical(k_key, logits[:, None], -1, (B, N)) # (B, N)

    eps = random.normal(e_key, (B, N, d)) # (B, N, d)

    means_k = jnp.take_along_axis(means, ks[..., None], axis=1) # (B, N, d)
    Ls_k = jnp.take_along_axis(Ls, ks[..., None, None], axis=1) # (B, N, d, d)

    y = means_k + jnp.einsum("bnij, bnj -> bni", Ls_k, eps) # (B, N, d)

    return y


def gmm_marg_cdf_batch(y, # (batch, K, L_mc, d)
                 pis, # (batch, K)
                 means, # (batch, K, d)
                 Ls # (batch, K, d, d)
                 ):
    stds = jnp.sqrt(jnp.sum(Ls**2, axis=-1))  # (batch, K, d)

    y_exp    = y[:, :, :, None, :]            # (batch, K, L_mc, 1, d)
    means_exp= means[:, None, None, :, :]     # (batch, 1, 1, K, d)
    std_exp  = stds[:, None, None, :, :]       # (batch, 1, 1, K, d)

    z = (y_exp - means_exp) / std_exp         # (batch, K, L_mc, K, d)
    Phi = norm.cdf(z)                  # (batch, K, L_mc, K, d)

    pi_exp = pis[:, None, None, :, None]       # (batch, 1, 1, K, 1)
    F = jnp.sum(pi_exp * Phi, axis=3)         #  (batch, K, L_mc, d)
    return F

def gmm_marg_cdf(y, # (batch, N, d)
                 logits, # (batch, K)
                 means, # (batch, K, d)
                 chol_pars # (batch, K, d, d)
                 ):
    
    B, K, d = means.shape
    pis = softmax(logits, axis=-1)  # (batch, K)
    Ls = chol_pars_to_L(chol_pars, d)
    stds = jnp.sqrt(jnp.sum(Ls**2, axis=-1))  # (batch, K, d)

    y_exp    = y[:, :, None, :]            # (batch, N, 1, d)
    means_exp= means[:, None, :, :]        # (batch, 1, K, d)
    std_exp  = stds[:, None, :, :]       # (batch, 1, K, d)

    z = (y_exp - means_exp) / std_exp         # (batch, N, K, d)
    Phi = norm.cdf(z)                  # (batch, N, K, d)

    pi_exp = pis[:, None, :, None]       # (batch, 1, K, 1)
    F = jnp.sum(pi_exp * Phi, axis=2)         #  (batch, N, d)
    return F


def ED(u, # (batch, d)
       v, # (batch, K, L_mc, d)
       pis, # (batch, K)
       ):
    B, K, L_mc, d = v.shape
    M = K * L_mc # we will work with this new index to avoid a redundant axis
    v = jnp.reshape(v, (B, M, d)) # (batch, M, d)
    duv = jnp.linalg.norm(u[:, None]-v, axis=-1) # (batch, M)

    # need the weight L_mc times
    w = jnp.repeat(pis / L_mc, repeats=L_mc, axis=1)   # (batch, M)

    _A = jnp.sum(w * duv, axis=-1) # (batch)

    i_idx, j_idx = jnp.triu_indices(M, k=1) # upper triangular indices 

    dvv = jnp.linalg.norm(v[:, i_idx] - v[:, j_idx], axis=-1) # (batch, P), P = (M^2-M)/2

    wij =  w[:, i_idx] * w[:, j_idx] # (batch, P)

    # renormalize for removing diagonal
    _B = jnp.sum(wij * dvv, axis=-1) / jnp.sum(wij, axis=-1) # (batch)

    ed = jnp.mean(2*_A - _B)

    return ed # ()


# jit and free up buffer of state since it will not be used anymore after this call 
@partial(jax.jit, donate_argnames="state", static_argnames="L_mc")
def train_step(eps_key: random.PRNGKey,
               state: train_state.TrainState,
               u: jnp.ndarray, # (batch, d)
               x: jnp.ndarray, # (batch, x_dim)
               L_mc: int,
              ) -> Tuple[train_state.TrainState, jnp.ndarray]:

    def loss_fn(params):

        logits, means, chol_pars = state.apply_fn(params, x)
        B, K, d = means.shape
        Ls = chol_pars_to_L(chol_pars, d) # (batch, K, d, d)
        pis = softmax(logits, axis=-1)

        # sample learned copula
        eps = random.normal(eps_key, (B, K, L_mc, d))
        y = means[:, :, None, :] + jnp.einsum("bkij, bkmj->bkmi", Ls, eps) # (batch, K, L_mc, d)
        v = gmm_marg_cdf_batch(y, pis, means, Ls) # (batch, K, L_mc, d)

        ed = ED(u, v, pis)

        return ed

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def create_train_state(key: Any, model: GMM,
                       learning_rate: float,
                       x_shape: Tuple[int,int]
                      ) -> train_state.TrainState:
    
    """Initial training state, will be updated in `train_step`"""

    params = model.init(key, jnp.zeros(x_shape)) # model is stateless, not affected by calling init
    tx     = optax.adam(learning_rate)
    # tx = optax.sgd(learning_rate, momentum=0.9)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def train_GMM_generator(keys: map,
                    model: GMM,
                    u: jnp.ndarray,
                    x: jnp.ndarray,
                    learning_rate: float,
                    n_epochs: int,
                    batch_size: int,
                    L_mc: int,
                    # u_val: jnp.ndarray | None = None,
                    # x_val: jnp.ndarray | None = None,
                    # early_stop: int | None = None
                    ):
    
    N, x_dim = x.shape


    losses = []

    state = create_train_state(next(keys),
                               model, learning_rate, 
                               (batch_size, x_dim)
                               )
    n_batches = N // batch_size
    assert n_batches > 0

    t0 = time.perf_counter()
    for epoch in range(n_epochs):

        x_idc = random.permutation(next(keys), N)
        x = x[x_idc]
        u = u[x_idc]
       
        for b in range(0, n_batches):

            x_batch = x[b*batch_size : (b+1)*batch_size]
            u_batch = u[b*batch_size : (b+1)*batch_size]

            # if x_val is not None:
            #     val_loss = val_loss_fn(state=state, u = u_val, x = x_val, z = z_val)
            #     val_losses.append(val_loss)

            state, loss = train_step(eps_key=next(keys), # eps_key=next(keys),
                                     state=state, u=u_batch, x=x_batch, L_mc=L_mc)
            losses.append(loss)
        
        # if len(val_losses) > early_stop and val_loss > max(val_losses[-early_stop:-1]):
        #     print("\nearly stop\n")
        #     break

        if epoch % 10 == 0:
            # if x_val is not None:
            #     print(f"Epoch {epoch:03d}  train loss={loss:.4f}  val loss={val_loss:.4f}")
            # else:
            print(f"Epoch {epoch:03d}  train loss={loss:.4f}")
        
    t1 = time.perf_counter()

    print(f"Training took {(t1-t0):.2f}s")

    return (state, losses) # if x_val is None else (state, losses, val_losses)



if __name__ == "__main__":

    root_key = random.key(3)
    keys = map(partial(random.fold_in, root_key), itertools.count())

    path = "gmm_gen/"

    N = 1000
    d = 3
    n_epochs = 100

    batch_size = 1000
    L_mc = 50
    
    u = random.uniform(next(keys), (N, d))
    x = random.normal(next(keys), (N, d))

    # u_val = random.uniform(next(keys), (N_val, d))
    # x_val = random.normal(next(keys), (N_val, d))

    model = GMM(hidden_dims=2*[8], K=2, d=d)

    state, losses = train_GMM_generator(keys, 
                                        model, 
                                        u, x,  
                                        1e-3, n_epochs, batch_size, L_mc
                                        )
    save_gen(path, state.params)
    
    plot_loss(losses, path)

    
