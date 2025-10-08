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
import math
jax.config.update("jax_enable_x64", True)


def sig_matrix(y, alpha):

    """
    Compute sig matrix

    Args
        y (jnp.array): (batch, 1)
        alpha: ()

    Returns
        log_mat (jnp.array): (batch, batch)
    """

    
    s_diff = alpha * (y - y.T)
    log_mat = jax.nn.sigmoid(s_diff)

    return log_mat

def sig_ecdf_vals(y, alpha):
    """
    Compute differentiable ECDF values (not sorted!)

    Args
        y (jnp.array): (batch, 1)
        alpha: ()

    Returns
        ecdf (jnp.array): (batch, 1)
    """

    log_mat = sig_matrix(y=y, alpha=alpha) # (batch, batch)

    ecdf = log_mat.mean(axis=-1, keepdims=True)

    return ecdf

if __name__ == "__main__":
    key = random.PRNGKey(1)
    n = 5000
    y = random.normal(key, (n,1))

    ecdf = sig_ecdf_vals(y, alpha=10)
    print(y)
    print(ecdf)

    plt.scatter(y, ecdf, s=1)
    plt.show()