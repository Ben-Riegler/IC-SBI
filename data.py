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
from mdn import MDN

jax.config.update("jax_enable_x64", True)


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

    return x.squeeze(-1), θ.squeeze(-1)

def mvn_posterior(x, prior_mean, prior_L, model_L):
    """
    Compute posterior parameters given data x
    
    Args
        x: shape (batch, d)
        prior_mean: shape (d, 1)
        prior_L: shape (d, d)
        model_L: shape (d, d)

    Returns
        post_mean: shape (batch, d)
        post_var: shape (d, d)
    """
    
    LL0 = jnp.matmul(prior_L, prior_L.T) # (d, d)
    LL1 = jnp.matmul(model_L, model_L.T) 
    Varx = LL0 + LL1
    Varx_inv = jnp.linalg.inv(Varx)
    Covθx = LL0
    B = jnp.linalg.matmul(Covθx, Varx_inv)
    post_mean = prior_mean + jnp.linalg.matmul(B, x.T-prior_mean) # (d, batch)
    post_var = LL0 - jnp.linalg.matmul(B, Covθx) # (d, d)

    return post_mean.T, post_var
