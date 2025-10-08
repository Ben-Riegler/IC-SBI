import jax
from jax import random
import matplotlib.pyplot as plt
import time

jax.config.update("jax_enable_x64", True)


@jax.jit
def sig_ecdf_vals(y, alpha):
    """
    Compute differentiable ECDF values (not sorted!)

    Args
        y (jnp.array): (batch, 1)
        alpha (jnp.array): (1, 1)

    Returns
        ecdf (jnp.array): (batch, 1)
    """

    s_diff = alpha * (y - y.T)
    log_mat = jax.nn.sigmoid(s_diff)

    ecdf = log_mat.mean(axis=-1, keepdims=True)

    return ecdf

if __name__ == "__main__":
    
    key = random.PRNGKey(1)
    n = 5000
    y = random.normal(key, (n,1))

    t0 = time.perf_counter()
    ecdf = sig_ecdf_vals(y, alpha=10)
    t1 = time.perf_counter()
    print(f"{(t1-t0):.2f}")

    # print(y)
    # print(ecdf)

    plt.scatter(y, ecdf, s=1)
    plt.show()
