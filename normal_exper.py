import os
from datetime import datetime
import pprint
import jax
import jax.numpy as jnp
from jax import random
from jax import device_get
import numpy as np

from mdn import MDN, train_marginal_mdns, get_cdf_vals
from gen import generator, train_generator
from data import gen_mv_normal_normal_data, mvn_posterior, sample_x_marginal, get_true_cdf, get_true_quantiles
from plots import plot_loss, plot_losses, plot_mvn_marginals, plot_mdn_marginals, plot_post_pairs, plot_marginal_hists, plot_marginals_and_mdn
from utils import save_gen
from mdn_inv import mdn_inv_marg
from ecdf import sig_marg_ecdf_vals, marg_ecdf_vals
from utils import str2bool

import argparse

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
os.environ["JAX_TRACEBACK_FILTERING"] = "0"

parser = argparse.ArgumentParser()

# prelim
parser.add_argument('--folder', type=str, default="normal_test15/")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--comment", type=str, default="No comment")

# data
parser.add_argument('--d', type=int, default=3)
parser.add_argument('--N', type=int, default=10000)

# MDNs set up
parser.add_argument('--mdn_lr', type=float, default=1e-3)

# post MDNs
parser.add_argument("--post_learn_cdf", type=str2bool, default="True")
parser.add_argument('--post_mdn_K', type=int, default=3)
parser.add_argument('--post_mdn_hidden',  nargs='+', type=int, default=3 * [8])
parser.add_argument('--post_mdn_epochs', type=int, default=200)
parser.add_argument('--post_mdn_batch_size', type=int, default=5000)


# generator architecture
parser.add_argument("--gen_emb_dim", type=int, default=32)
parser.add_argument('--gen_hidden_dims',  nargs='+', type=int, default=[128])

# generator training
parser.add_argument("--gen_epochs", type=int, default=50)
parser.add_argument("--gen_batch_size", type=int, default=10000)
parser.add_argument('--gen_lr', type=float, default=1e-3)
parser.add_argument("--gen_z_batch_size", type=int, default=50)

# fit lat MDNs
parser.add_argument("--lat_learn_cdf", type=str2bool, default="False")
parser.add_argument("--lat_mdn_N_fit", type=int, default=30000)
parser.add_argument('--lat_mdn_K', type=int, default=3)
parser.add_argument('--lat_mdn_hidden',  nargs='+', type=int, default=3 * [8])
parser.add_argument('--lat_mdn_epochs', type=int, default=200)
parser.add_argument('--lat_mdn_batch_size', type=int, default=10000)

# test
parser.add_argument("--N_test", type=int, default=5000)

args = parser.parse_args()

# set up and data
path = "experiments/" + args.folder
os.makedirs(path, exist_ok=True)

# save config
arg_dict = vars(args)
comm = arg_dict["comment"]
arg_dict.pop("comment")
with open(os.path.join(path, "config.txt"), "w", encoding="utf-8") as f:
    f.write(str(datetime.now()))
    f.write("\n\n")
    f.write(f"Comment: {comm}\n\n")
    f.write(pprint.pformat(arg_dict, width=100))
 
d = args.d
N = args.N

post_learn_cdf = args.post_learn_cdf
post_mdn_K = args.post_mdn_K
post_mdn_hidden = args.post_mdn_hidden
post_mdn_batch_size = args.post_mdn_batch_size
post_mdn_epochs = args.post_mdn_epochs

mdn_lr = args.mdn_lr

gen_emb_dim = args.gen_emb_dim
gen_hidden_dims = args.gen_hidden_dims
gen_epochs = args.gen_epochs
gen_batch_size = args.gen_batch_size
gen_z_batch_size = args.gen_z_batch_size
gen_lr = args.gen_lr
Nz = gen_z_batch_size * N // gen_batch_size

lat_learn_cdf = args.lat_learn_cdf
lat_mdn_N_fit = args.lat_mdn_N_fit
lat_mdn_hidden = args.lat_mdn_hidden
lat_mdn_K = args.lat_mdn_K
lat_mdn_batch_size = args.lat_mdn_batch_size
lat_mdn_epochs = args.lat_mdn_epochs

key = random.PRNGKey(args.seed)
key, k1, k2, k3, k4, k5, k_data = random.split(key, 7)

L0 = jnp.sqrt(0.1) * jnp.eye(d)
L1 = jnp.sqrt(0.1) * jnp.eye(d)

# L0 = random.normal(k1, (d,d))
# L1 = random.normal(k2, (d,d))

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
    post_mdn_hidden=device_get(post_mdn_hidden),   
    lat_mdn_hidden=device_get(lat_mdn_hidden),           
    gen_hidden_dims=device_get(gen_hidden_dims),
    emb_dim=device_get(gen_emb_dim),
    post_mdn_K=device_get(post_mdn_K),
    lat_mdn_K=device_get(lat_mdn_K)
)

z = random.normal(k4, (Nz, d))

print("\n----------Generating data----------\n")
x_data, θ_data = gen_mv_normal_normal_data(k_data, 
                                            n_samples=N, 
                                            prior_mean=prior_mean,
                                            prior_L=L0, 
                                            model_L=L1)

post_mean, post_var = mvn_posterior(x_data, prior_mean, L0, L1)

print("x: ", x_data.shape, 
      "theta: ", θ_data.shape, 
      "post means: ", post_mean.shape, 
      "post var", post_var.shape)


if post_learn_cdf:
    print("\n----------Train posterior MDNs----------\n")

    post_mdn = MDN(hidden_dims = post_mdn_hidden,
            K = post_mdn_K)

    losses_list, post_par_list = train_marginal_mdns(k2, 
                                                model=post_mdn, x_data=x_data, θ_data=θ_data, 
                                                lr=mdn_lr, n_epochs=post_mdn_epochs, batch_size=post_mdn_batch_size, 
                                                path=path+"post_mdn/")
    plot_losses(losses_list, path+"post_mdn/")

    test_ids = random.choice(k3, N, (4,))
    x_test = x_data[test_ids]
    plot_mvn_marginals(post_mdn, post_par_list, x_test, prior_mean, L0, L1, theta_range=(-5, 5), path = path + "post_mdn/")

    u = get_cdf_vals(model=post_mdn, par_list=post_par_list, 
                    x_data=x_data,θ_data=θ_data)
    
else:
    u = get_true_cdf(theta=θ_data, post_mean=post_mean, post_var=post_var)


print("\n----------Train generator----------\n")

gen = generator(emb_dim=gen_emb_dim, hidden_dims=gen_hidden_dims, out_dim=d)

gen_state, losses = train_generator(k5,
                                model=gen,
                                u=u, x=x_data, z=z, 
                                learning_rate=gen_lr, 
                                n_epochs=gen_epochs, batch_size=gen_batch_size, z_batch_size=gen_z_batch_size)

save_gen(path, gen_state.params)

plot_loss(losses, path+"gen/")

if lat_learn_cdf:
    print("\n----------Train latents MDNs----------\n")

    # fit MDNs to marginals F(y_j|x) for learned p(y|x)
    # sample p(y,x) = p(x)p(y|x)
    # x_data ~ p(x) or oracle DGP

    key, z_key, x_key = random.split(key, 3)

    z_samples = random.normal(z_key, (lat_mdn_N_fit, d))
    x_samples = sample_x_marginal(x_key, lat_mdn_N_fit, prior_mean, L0, L1)
    y_samples = gen.apply(gen_state.params, z=z_samples, x=x_samples) # ~ p(y|x) (N_fit, d)

    lat_mdn = MDN(hidden_dims = lat_mdn_hidden,
            K = lat_mdn_K)

    key, y_key = random.split(key)
    losses_list, y_par_list = train_marginal_mdns(y_key, 
                                                model=lat_mdn, x_data=x_samples, θ_data=y_samples, 
                                                lr=mdn_lr, n_epochs=lat_mdn_epochs, batch_size=lat_mdn_batch_size, 
                                                path=path+"y_mdn/")

    plot_losses(losses_list, path+"y_mdn/")

print("\n----------Test model----------\n")
# test: posterior sampling
test_locs = 4
N_test = args.N_test

key, t_key = random.split(key)
test_ids = random.choice(t_key, N, (test_locs,))
x_test = x_data[test_ids] # (test_locs, d)
x_test_exp = jnp.repeat(x_test[:, None, :], axis=1, repeats=N_test)

key, z_key = random.split(key)

z_samples = random.normal(z_key, (test_locs, N_test, d))
y_samples = gen.apply(gen_state.params, z=z_samples, x=jnp.repeat(x_test[:, None, :], repeats = N_test, axis = 1 )) # (test_locs, N_test, d) each test_loc gets its own N_test samples

plot_marginal_hists(y_samples, x_test, save_path=path + "y_mdn/", title=rf"$p(y_j \mid x)$", name="y_marg_hist.pdf")

plot_post_pairs(
    theta=y_samples,
    x_vals=x_test,
    save_dir=os.path.join(path, "y_mdn/pairplots"),
    file_prefix="bivariate_samples",
    ax_lab=rf"y",
)

# map into learned copula space
if lat_learn_cdf:
    print("Using learned MDNs to map into learned copula space")
    v = get_cdf_vals(model=lat_mdn, par_list=y_par_list, x_data=x_test_exp, θ_data=y_samples) # (test_locs, N_test, d)
    plot_marginals_and_mdn(model=lat_mdn, params=y_par_list, x_vals=x_test, sample=y_samples, x_lab=rf"$y_j$",
                    path= path + "y_mdn/")
   
else:
    print("Using ECDF to map into learned copula space")
    v = marg_ecdf_vals(y_samples)
    
# should be uniform if MDNs learned correctly
plot_marginal_hists(v, x_test, save_path= path + "y_mdn/", name="marg_y_cop_hist.pdf")

# map into parameter space

if post_learn_cdf:
    print("Using learned marginal posteriors to map into parameter space")
    theta = mdn_inv_marg(model = post_mdn, par_list=post_par_list, x=x_test_exp, u=v) # (test_locs, N_test, d)
else:
    print("Using true quantile function to map into parameter space")
    theta = get_true_quantiles(u=v, x = x_test, prior_mean=prior_mean, prior_L=L0, model_L=L1)

# Ground truth posterior at x_test
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

print("\npairplots saved\n")
print("\ndone!")

