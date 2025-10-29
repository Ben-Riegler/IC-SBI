import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

jax.config.update("jax_enable_x64", True)


# @jax.jit
# def sig_ecdf_vals(y, alpha = jnp.array(10)):
#     """
#     Compute differentiable ECDF values (not sorted!)

#     Args
#         y (jnp.array): (batch, 1)
#         alpha (jnp.array): (1, 1)

#     Returns
#         ecdf (jnp.array): (batch, 1)
#     """

#     s_diff = alpha * (y - y.T)
#     log_mat = jax.nn.sigmoid(s_diff)

#     ecdf = log_mat.mean(axis=-1, keepdims=True)

#     return ecdf

@jax.jit
def sig_marg_ecdf_vals(y, alpha = jnp.array(10)):
    """
    Compute marginal differentiable ECDF values (not sorted!)

    Args
        y (jnp.array): (batch, z_batch, y_dim)
        alpha (jnp.array): (1, 1)

    Returns
        ecdf (jnp.array): (batch, z_batch, y_dim)
    """

    y = jnp.swapaxes(y, 1, 2) # (batch, y_dim, z_batch)
    s_diff = alpha * (y[..., None] - y[..., None, :]) # (batch, y_dim, z_batch, z_batch)
    log_mat = jax.nn.sigmoid(s_diff)

    ecdf = log_mat.mean(axis=-1) # (batch, y_dim, z_batch)

    return jnp.swapaxes(ecdf, 1, 2) # (batch, z_batch, y_dim)

# @jax.jit
def marg_ecdf_vals(y):
    """
    Compute marginal ECDF values (not sorted!)

    Args
        y (jnp.array): (batch, z_batch, y_dim)
        alpha (jnp.array): (1, 1)

    Returns
        ecdf (jnp.array): (batch, z_batch, y_dim)
    """

    y = jnp.swapaxes(y, 1, 2) # (batch, y_dim, z_batch)
    idcs = jnp.argsort(y, axis=-1)
    ranks = jnp.argsort(idcs, axis=-1)

    ecdf = (ranks+1) / y.shape[-1]
    return jnp.swapaxes(ecdf, 1, 2) # (batch, z_batch, y_dim)
    


if __name__ == "__main__":
    
    key = random.PRNGKey(1)
    n = 5000
    y = random.normal(key, (1, n, 2))

    t0 = time.perf_counter()
    ecdf = sig_marg_ecdf_vals(y, alpha=10)

    ecdf2 = marg_ecdf_vals(y)

    t1 = time.perf_counter()
    print(f"{(t1-t0):.2f}")

    plt.scatter(y[0,:, 0], ecdf[0,:, 0], s=1)
    plt.show()
    plt.hist(ecdf[0,:, 0])
    plt.show()
    
    plt.scatter(y[0,:, 0], ecdf2[0,:, 0], s=1)
    plt.show()
    plt.hist(ecdf2[0,:, 0])
    plt.show()
