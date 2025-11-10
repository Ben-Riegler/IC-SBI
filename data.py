
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm

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

def sample_x_marginal(key, n_samples, prior_mean, prior_L, model_L):
    d = prior_L.shape[0]
    Sigma_x = prior_L @ prior_L.T + model_L @ model_L.T
    Lx = jnp.linalg.cholesky(Sigma_x) # (d, d)
    eps = random.normal(key, (n_samples, d, 1))
    x = prior_mean + jnp.matmul(Lx, eps)
    return x.squeeze(-1)  # (n_samples, d)


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



def get_true_cdf(theta, # (batch, d)
                 post_mean, # (batch, d)
                 post_var # (d, d)
                 ):

    z = (theta - post_mean) / jnp.sqrt(jnp.diag(post_var))[None, :]
    u = norm.cdf(z)

    return u


def get_true_quantiles(u, # (test_batch, batch, d)
                       x, # (test_batch, x_dim)
                       prior_mean, prior_L, model_L
                       ):
    
    post_means, post_var = mvn_posterior(x, prior_mean, prior_L, model_L) # (test_batch, d), (d, d)

    post_std = jnp.sqrt(jnp.diag(post_var))[None, :][None, ...] # (1, 1, d)

    t = post_means[:, None] + post_std * norm.ppf(u) 

    return t # (test_batch, batch, d)


def gen_two_moons_data(key, n_samples, prior_low, prior_high):
    k_theta, k_a, k_r = random.split(key, 3)

    # Prior over parameters θ ~ Uniform(prior_low, prior_high)
    low = jnp.asarray(prior_low)
    high = jnp.asarray(prior_high)
    d = low.shape[-1]
    theta = low + (high - low) * random.uniform(k_theta, (n_samples, d))

    # Simulator hyperparameters
    a_low = -jnp.pi / 2.0
    a_high = +jnp.pi / 2.0
    base_offset = 0.25
    r_loc = 0.1
    r_scale = 0.01

    # Latent polar variables
    a = random.uniform(k_a, (n_samples,)) * (a_high - a_low) + a_low
    r = r_loc + r_scale * random.normal(k_r, (n_samples,))

    # Base point on the small arc
    p_x = jnp.cos(a) * r + base_offset
    p_y = jnp.sin(a) * r
    p = jnp.stack([p_x, p_y], axis=-1)

    # Affine, angle-dependent map with rotation by ang = -π/4
    ang = -jnp.pi / 4.0
    c = jnp.cos(ang)
    s = jnp.sin(ang)
    z0 = c * theta[:, 0] - s * theta[:, 1]
    z1 = s * theta[:, 0] + c * theta[:, 1]

    x = p + jnp.stack([-jnp.abs(z0), z1], axis=-1)
    return x, theta


def sample_two_moons_posterior(key, x, n_samples, prior_low, prior_high):
    """

    Args:
        key: jax.random.PRNGKey
        x: array-like, shape (2,), observed data
        n_samples: int, number of posterior samples to draw
        prior_low: array-like, shape (2,), lower bounds of Uniform prior
        prior_high: array-like, shape (2,), upper bounds of Uniform prior

    Returns:
        theta: jnp.ndarray, shape (n_samples, 2), posterior samples of θ
    """
    x = jnp.asarray(x).reshape(2,)
    low = jnp.asarray(prior_low).reshape(2,)
    high = jnp.asarray(prior_high).reshape(2,)

    a_low = -jnp.pi / 2.0
    a_high = +jnp.pi / 2.0
    base_offset = 0.25
    r_loc = 0.1
    r_scale = 0.01

    c = jnp.cos(jnp.pi / 4.0)
    s = jnp.sin(jnp.pi / 4.0)

    def draw_batch(k, m):
        ka, kr, ks = random.split(k, 3)
        a = random.uniform(ka, (m,), minval=a_low, maxval=a_high)
        r = r_loc + r_scale * random.normal(kr, (m,))


        p_x = jnp.cos(a) * r + base_offset
        p_y = jnp.sin(a) * r

        q0 = p_x - x[0]
        q1 = x[1] - p_y
        sign = jnp.where(random.uniform(ks, (m,)) < 0.5, -1.0, 1.0)
        z0 = sign * q0
        z1 = q1

        theta1 = c * z0 - s * z1
        theta2 = s * z0 + c * z1
        theta = jnp.stack([theta1, theta2], axis=1)


        mask = jnp.all((theta >= low) & (theta <= high), axis=1)
        return theta, mask


    batch_size = max(4 * int(n_samples), 4096)
    collected = []
    total = 0
    k = key

    while total < n_samples:
        k, kb = random.split(k)
        theta_b, mask_b = draw_batch(kb, batch_size)
        kept = theta_b[mask_b]
        if kept.size > 0:
            collected.append(kept)
            total += kept.shape[0]

    theta_all = jnp.concatenate(collected, axis=0)[:n_samples]
    return theta_all

