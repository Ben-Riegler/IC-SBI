import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from typing import List, Any, Tuple
import matplotlib.pyplot as plt
from jax.scipy.stats import norm

from mdn import MDN, train_marginal_mdns
from gen import generator, train_generator
from data import gen_mv_normal_normal_data, mvn_posterior
from plots import plot_loss, plot_losses, plot_mvn_marginals, plot_mdn_marginals
from utils import save_gen


# set up and data
path = "normal/"
os.makedirs(path, exist_ok=True)

d = 4
K = 4
N = 5000
mdn_batch_size = 5000
mdn_epochs = 5000
mdn_lr = 1e-4
mdn_hidden_dims = 2 * [8]

emb_dim = 8
gen_hidden_dims = 3 * [8]
gen_epochs = 5000
gen_batch_size = 5000
z_batch_size = 20
gen_lr = 1e-4
Nz = z_batch_size * N // gen_batch_size



key = random.PRNGKey(1)
key, k1, k2, k3, k4, k5 = random.split(key, 6)
L0 = random.normal(k1, (d,d))
L1 = random.normal(k2, (d,d))

prior_mean = jnp.zeros((d,1))

z = random.normal(k4, (Nz, d))

x_data, θ_data = gen_mv_normal_normal_data(k1, 
                                            n_samples=N, 
                                            prior_mean=prior_mean,
                                            prior_L=L0, 
                                            model_L=L1)

post_mean, post_var = mvn_posterior(x_data, prior_mean, L0, L1)

print(x_data.shape, θ_data.shape, post_mean.shape, post_var.shape)

mdn = MDN(hidden_dims = mdn_hidden_dims,
          K = K)

losses_list, par_list = train_marginal_mdns(k2, 
                                            model=mdn, x_data=x_data, θ_data=θ_data, 
                                            lr=mdn_lr, n_epochs=mdn_epochs, batch_size=mdn_batch_size, 
                                            path=path+"mdn/")
plot_losses(losses_list, path+"mdn/")

test_ids = random.choice(k3, N, (4,))
x_test = x_data[test_ids]
plot_mvn_marginals(mdn, par_list, x_test, prior_mean, L0, L1, theta_range=(-5, 5), path = path + "mdn/")

def get_cdf_vals(model, par_list, x_data, θ_data):

    u = []
    # Loop over dimensions
    for dim in range(d):

        logits, means, log_scales = model.apply(par_list[dim], x_data)  # each: (batch, K)
        scales = jnp.exp(log_scales) 
        log_pi = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)  # (batch, K)
        pi = jnp.exp(log_pi)                                           # (batch, K)

        # Learned mixture CDF
        comp_cdfs = norm.cdf((θ_data[:, dim][:, None] - means) / scales)  # (batch, K)
        u_ = jnp.sum(pi * comp_cdfs, axis=-1) # (batch)
        u.append(u_)

    u = jnp.column_stack(u)

    return u

u = get_cdf_vals(model=mdn, par_list=par_list, 
                 x_data=x_data,θ_data=θ_data)

gen = generator(emb_dim=emb_dim, hidden_dims=gen_hidden_dims, out_dim=d)

state, losses = train_generator(k5,
                                model=gen,
                                u=u, x=x_data, z=z, 
                                learning_rate=gen_lr, 
                                n_epochs=gen_epochs, batch_size=gen_batch_size, z_batch_size=z_batch_size)

save_gen(path, state.params)

plot_loss(losses, path+"gen/")

# fit MDNs to marginals F(y_j|x) for learned P(Y|x)
# sample p(y,x) = p(x)p(y|x)
# x_data ~ p(x)

key, z_key = random.split(key)

z_samples = random.normal(z_key, (N, d))
y_samples = gen.apply(state.params, z=z_samples, x=x_data) # ~ p(y|x)

mdn = MDN(hidden_dims = mdn_hidden_dims,
          K = K)

key, y_key = random.split(key)
losses_list, par_list = train_marginal_mdns(y_key, 
                                            model=mdn, x_data=x_data, θ_data=y_samples, 
                                            lr=mdn_lr, n_epochs=mdn_epochs, batch_size=mdn_batch_size, 
                                            path=path+"y_mdn/")

plot_losses(losses_list, path+"y_mdn/")

key, t_key = random.split(key)
test_ids = random.choice(t_key, N, (4,))
x_test = x_data[test_ids]
plot_mdn_marginals(model=mdn, params=par_list, theta_range=(-5, 5), path= path + "y_mdn/")
