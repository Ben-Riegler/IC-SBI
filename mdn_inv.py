import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

def mixture_cdf(y, pi, mu, sigma):
    """F(y) for a Gaussian mixture. Shapes:
       y: (...,)
       pi, mu, sigma: (..., K)
       returns: (...,)
    """
    t = (y[..., None] - mu) / sigma           # (..., K)
    return jnp.sum(pi * norm.cdf(t), axis=-1)     # (...,)

def mixture_moments(pi, mu, sigma, eps=1e-12):
    """Return mean and std of the mixture (…,) (used to build brackets)."""
    m  = jnp.sum(pi * mu, axis=-1)
    s2 = jnp.sum(pi * (sigma**2 + mu**2), axis=-1) - m**2
    s2 = jnp.maximum(s2, eps)
    return m, jnp.sqrt(s2)

@jax.jit
def mixture_quantile_bisect(u, pi, mu, sigma, iters=40, L=10.0):
    """Invert F(y)=u via batched bisection. All shapes broadcast on leading axes.
       u: (...,)
       pi, mu, sigma: (..., K)
       returns y: (...,)
    """
    # keep u away from exact 0/1
    u = jnp.clip(u, 1e-12, 1 - 1e-12)

    # build wide brackets per sample using mixture mean±L*std, widened by max sigma
    m, s = mixture_moments(pi, mu, sigma)
    maxsig = jnp.max(sigma, axis=-1)
    lo = m - L * (s + maxsig)
    hi = m + L * (s + maxsig)

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        Fmid = mixture_cdf(mid, pi, mu, sigma)
        lo = jnp.where(Fmid < u, mid, lo)
        hi = jnp.where(Fmid >= u, mid, hi)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=iters)
    return 0.5 * (lo + hi)


def mdn_params_for_x(model, params, x):
    """Apply MDN to x and return pi, mu, sigma. Shapes:
       x: (..., x_dim)  →  pi, mu, sigma: (..., K)
    """
    
    logits, means, log_scales = model.apply(params, x)  # (..., K) each
    log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)  # (..., K)
    pi = jnp.exp(log_pi)             # (..., K)
    sigma  = jnp.exp(log_scales)           
    return pi, means, sigma

def mdn_inv_marg(model, par_list, x, u):
    """Invert MDN marginals for all dims.
       x: (..., x_dim)
       u: (..., d)  with u in (0,1)
       par_list: list of MDN params per marginal dim, length d
       returns y: (..., d)
    """
    d = u.shape[-1]
    ys = []
    for j in range(d):
        pi, mu, sigma = mdn_params_for_x(model, par_list[j], x)   # (..., K)
        yj = mixture_quantile_bisect(u[..., j], pi, mu, sigma)    # (...,)
        ys.append(yj)
    return jnp.stack(ys, axis=-1)                                  # (..., d)


def multi_mdn_inv_marg(model, params, x, u):

    """Invert MDN marginals for all dims.
       x: (..., x_dim)
       u: (..., d)  with u in (0,1)
       params: MDN params 
       returns y: (..., d)
    """
    pi, mu, sigma = mdn_params_for_x(model, params, x) # each (..., d, K)
    y = jax.vmap(mixture_quantile_bisect, in_axes=(-1, -2, -2, -2), out_axes=-1)(u, pi, mu, sigma)

    return y