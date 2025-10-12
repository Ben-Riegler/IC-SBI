import os
import jax.numpy as jnp
from jax import random
from jax import device_get
import numpy as np

from mdn import MDN, train_marginal_mdns, get_cdf_vals
from gen import generator, train_generator
from data import gen_mv_normal_normal_data, mvn_posterior, sample_x_marginal
from plots import plot_loss, plot_losses, plot_mvn_marginals, plot_mdn_marginals, plot_post_pairs
from utils import save_gen
from mdn_inv import mdn_inv_marg

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--folder', type=str, default="results")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument('--d', type=int, default=2)

parser.add_argument('--K', type=int, default=2)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--mdn_batch_size', type=int, default=1)
parser.add_argument('--mdn_epochs', type=int, default=1)
parser.add_argument('--mdn_lr', type=float, default=1e-4)
parser.add_argument('--mdn_hidden_dims', type=list, default=2 * [8])

parser.add_argument("--emb_dim", type=int, default=8)
parser.add_argument('--gen_hidden_dims', type=list, default=5 * [8])
parser.add_argument("--gen_epochs", type=int, default=1)
parser.add_argument("--gen_batch_size", type=int, default=1)
parser.add_argument("--z_batch_size", type=int, default=1)
parser.add_argument('--gen_lr', type=float, default=1e-4)

parser.add_argument("--N_fit", type=int, default=100)
parser.add_argument("--N_test", type=int, default=100)

args = parser.parse_args()


# set up and data
path = args.folder
os.makedirs(path, exist_ok=True)

d = args.d
K = args.K
N = args.N
mdn_batch_size = args.mdn_batch_size
mdn_epochs = args.mdn_epochs
mdn_lr = args.mdn_lr
mdn_hidden_dims = args.mdn_hidden_dims

emb_dim = args.emb_dim

gen_hidden_dims = args.gen_hidden_dims
gen_epochs = args.gen_epochs
gen_batch_size = args.gen_batch_size
z_batch_size = args.z_batch_size
gen_lr = args.gen_lr
Nz = z_batch_size * N // gen_batch_size



key = random.PRNGKey(args.seed)
key, k1, k2, k3, k4, k5 = random.split(key, 6)
L0 = random.normal(k1, (d,d))
L1 = random.normal(k2, (d,d))

prior_mean = jnp.zeros((d,1))

# save for later
np.savez_compressed(
    os.path.join(path, "DGP.npz"),
    L0=device_get(L0),               # (d, d)
    L1=device_get(L1),               # (d, d)
    prior_mean=device_get(prior_mean)  # (d, 1)
)

np.savez_compressed(
    os.path.join(path, "Pars.npz"),
    mdn_hidden_dims=device_get(mdn_hidden_dims),             
    gen_hidden_dims=device_get(gen_hidden_dims),
    emb_dim=device_get(emb_dim),
    K=device_get(K)
)

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

losses_list, post_par_list = train_marginal_mdns(k2, 
                                            model=mdn, x_data=x_data, θ_data=θ_data, 
                                            lr=mdn_lr, n_epochs=mdn_epochs, batch_size=mdn_batch_size, 
                                            path=path+"post_mdn/")
plot_losses(losses_list, path+"post_mdn/")

test_ids = random.choice(k3, N, (4,))
x_test = x_data[test_ids]
plot_mvn_marginals(mdn, post_par_list, x_test, prior_mean, L0, L1, theta_range=(-5, 5), path = path + "post_mdn/")


u = get_cdf_vals(model=mdn, par_list=post_par_list, 
                 x_data=x_data,θ_data=θ_data)

gen = generator(emb_dim=emb_dim, hidden_dims=gen_hidden_dims, out_dim=d)

gen_state, losses = train_generator(k5,
                                model=gen,
                                u=u, x=x_data, z=z, 
                                learning_rate=gen_lr, 
                                n_epochs=gen_epochs, batch_size=gen_batch_size, z_batch_size=z_batch_size)

save_gen(path, gen_state.params)

plot_loss(losses, path+"gen/")

# fit MDNs to marginals F(y_j|x) for learned P(Y|x)
# sample p(y,x) = p(x)p(y|x)
# x_data ~ p(x)

N_fit = args.N_fit
key, z_key, x_key = random.split(key, 3)

z_samples = random.normal(z_key, (N_fit, d))
x_samples = sample_x_marginal(x_key, N_fit, prior_mean, L0, L1)
y_samples = gen.apply(gen_state.params, z=z_samples, x=x_samples) # ~ p(y|x) (N, d)

mdn = MDN(hidden_dims = mdn_hidden_dims,
          K = K)

key, y_key = random.split(key)
losses_list, y_par_list = train_marginal_mdns(y_key, 
                                            model=mdn, x_data=x_data, θ_data=y_samples, 
                                            lr=mdn_lr, n_epochs=mdn_epochs, batch_size=mdn_batch_size, 
                                            path=path+"y_mdn/")

plot_losses(losses_list, path+"y_mdn/")

# test: posterior sampling
test_locs = 4
N_test = args.N_test

key, t_key = random.split(key)
test_ids = random.choice(t_key, N, (test_locs,))
x_test = x_data[test_ids] # (test_locs, d)
plot_mdn_marginals(model=mdn, params=y_par_list, x_vals=x_test, theta_range=(-5, 5), path= path + "y_mdn/")

key, z_key = random.split(key)

z_samples = random.normal(z_key, (test_locs, N_test, d))
y_samples = gen.apply(gen_state.params, z=z_samples, x=x_test[:, None, :]) # (test_locs, N_test, d) each test_loc gets its own N_test samples


# map into learned copula space
x_test_exp = jnp.repeat(x_test[:, None, :], axis=1, repeats=N_test)
v = get_cdf_vals(model=mdn, par_list=y_par_list, x_data=x_test_exp, θ_data=y_samples) # (test_locs, N_test, d)



# map into parameter space
theta = mdn_inv_marg(model = mdn, par_list=post_par_list, x=x_test_exp, u=v) # (test_locs, N_test, d)

# Ground truth posterior at x_test
# If you already computed full post_mean/post_var for x_data, just index them:
mu_gt  = post_mean[test_ids]        
cov_gt = jnp.repeat(post_var[None, ...], axis=0, repeats=test_locs)                   # (test_locs, d, d)

plot_post_pairs(
    theta=theta,
    x_vals=x_test,
    post_mean=mu_gt,
    post_cov=cov_gt,
    save_dir=os.path.join(path, "pairplots"),
    file_prefix="posterior_pairs",
)
