import matplotlib.pyplot as plt
import os
import pprint
import jax
import jax.numpy as jnp
from jax import random
from jax import device_get
import numpy as np

from mdn import MDN, train_marginal_mdns, get_cdf_vals
from gen import generator, train_generator
from data import gen_two_moons_data, sample_two_moons_posterior
from plots import plot_loss, plot_losses, plot_mdn_marginals, plot_post_pairs, plot_marginal_cdf_hists
from utils import save_gen
from mdn_inv import mdn_inv_marg

import argparse

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

parser = argparse.ArgumentParser()

parser.add_argument('--folder', type=str, default="moon_test/")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument('--d', type=int, default=2)

parser.add_argument('--K', type=int, default=2)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--post_mdn_batch_size', type=int, default=100)
parser.add_argument('--post_mdn_epochs', type=int, default=100)
parser.add_argument('--lat_mdn_batch_size', type=int, default=20000)
parser.add_argument('--lat_mdn_epochs', type=int, default=200)
parser.add_argument('--mdn_lr', type=float, default=1e-4)
parser.add_argument('--mdn_hidden_dims',  nargs='+', type=int, default=2 * [8])

parser.add_argument("--emb_dim", type=int, default=8)
parser.add_argument('--gen_hidden_dims',  nargs='+', type=int, default=5 * [8])
parser.add_argument("--gen_epochs", type=int, default=1)
parser.add_argument("--gen_batch_size", type=int, default=1)
parser.add_argument("--z_batch_size", type=int, default=10)
parser.add_argument('--gen_lr', type=float, default=1e-4)

parser.add_argument("--N_fit", type=int, default=10000)
parser.add_argument("--N_test", type=int, default=100)

args = parser.parse_args()

# set up and data
path = "experiments/" + args.folder
os.makedirs(path, exist_ok=True)

# save args  
with open(os.path.join(path, "config.txt"), "w", encoding="utf-8") as f:
    f.write(pprint.pformat(vars(args), width=100))
 

d = args.d
K = args.K
N = args.N
post_mdn_batch_size = args.post_mdn_batch_size
post_mdn_epochs = args.post_mdn_epochs
lat_mdn_batch_size = args.lat_mdn_batch_size
lat_mdn_epochs = args.lat_mdn_epochs
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

p_low = d*[-1]
p_up = d*[1]

# save for later
np.savez_compressed(
    os.path.join(path, "DGP.npz"),
    p_low=device_get(p_low),               # (d,)
    p_up=device_get(p_up),               # (d,)
)

np.savez_compressed(
    os.path.join(path, "Pars.npz"),
    mdn_hidden_dims=device_get(mdn_hidden_dims),             
    gen_hidden_dims=device_get(gen_hidden_dims),
    emb_dim=device_get(emb_dim),
    K=device_get(K)
)

z = random.normal(k4, (Nz, d))
x_data, t_data = gen_two_moons_data(k1, N, prior_low=p_low, prior_high=p_up)
print("joint samples", x_data.shape, t_data.shape)

mdn = MDN(hidden_dims = mdn_hidden_dims,
          K = K)

losses_list, post_par_list = train_marginal_mdns(k2, 
                                            model=mdn, x_data=x_data, θ_data=t_data, 
                                            lr=mdn_lr, n_epochs=post_mdn_epochs, batch_size=post_mdn_batch_size, 
                                            path=path+"post_mdn/")
plot_losses(losses_list, path+"post_mdn/")

test_ids = random.choice(k3, N, (4,))
x_test = x_data[test_ids]

plot_mdn_marginals(model=mdn, 
                   params=post_par_list, 
                   x_vals=x_test, 
                   theta_range=(-1, 1), 
                   path = path + "post_mdn/")


u = get_cdf_vals(model=mdn, par_list=post_par_list, 
                 x_data=x_data,θ_data=t_data)

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
x_samples, _ = gen_two_moons_data(x_key, N_fit, prior_low=p_low, prior_high=p_up) # ~ p(x)
y_samples = gen.apply(gen_state.params, z=z_samples, x=x_samples) # ~ p(y|x) (N, d)
print(y_samples.shape)
mdn = MDN(hidden_dims = mdn_hidden_dims,
          K = K)

key, y_key = random.split(key)

losses_list, y_par_list = train_marginal_mdns(y_key, 
                                            model=mdn, x_data=x_samples, θ_data=y_samples, 
                                            lr=mdn_lr, n_epochs=lat_mdn_epochs, batch_size=lat_mdn_batch_size, 
                                            path=path+"y_mdn/")

plot_losses(losses_list, path+"y_mdn/")

# test: posterior sampling
test_locs = 4
N_test = args.N_test

key, t_key = random.split(key)
test_ids = random.choice(t_key, N, (test_locs,))
x_test = x_data[test_ids] # (test_locs, d)
plot_mdn_marginals(model=mdn, params=y_par_list, x_vals=x_test, theta_range=(-1, 1), path= path + "y_mdn/")

key, z_key = random.split(key)

z_samples = random.normal(z_key, (test_locs, N_test, d))
y_samples = gen.apply(gen_state.params, z=z_samples, x=x_test[:, None, :]) # (test_locs, N_test, d) each test_loc gets its own N_test samples

# map into learned copula space
x_test_exp = jnp.repeat(x_test[:, None, :], axis=1, repeats=N_test)
v = get_cdf_vals(model=mdn, par_list=y_par_list, x_data=x_test_exp, θ_data=y_samples) # (test_locs, N_test, d)

# should be uniform if MDNs learned correctly
plot_marginal_cdf_hists(v, x_test, save_path= path + "y_mdn/")

# map into parameter space
theta = mdn_inv_marg(model = mdn, par_list=post_par_list, x=x_test_exp, u=v) # (test_locs, N_test, d)

key, orc_key = random.split(key)
theta_true = jnp.stack([sample_two_moons_posterior(orc_key, 
                                                   x=x_test[i],
                                                   n_samples=N_test,
                                                   prior_low=p_low,
                                                   prior_high=p_up) for i in range(test_locs)], axis=0)
print(theta_true.shape)

plot_post_pairs(
    theta=theta,
    theta_true=theta_true,
    x_vals=x_test,
    save_dir=os.path.join(path, "pairplots"),
    file_prefix="posterior_pairs",
)


# key, te_key = random.split(key)
# test_idx = random.choice(te_key, N)

# x_test = x_data[test_idx]
# print(x_test.shape)

# plt.scatter(x_data[:,0], x_data[:,1])
# plt.show()

# key, p_key = random.split(key)
# t_post = sample_two_moons_posterior(p_key, x_test, N, prior_low=2*[-4], prior_high=2*[4])



