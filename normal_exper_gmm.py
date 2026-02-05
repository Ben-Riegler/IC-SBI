import os
from datetime import datetime
import pprint
import jax
import jax.numpy as jnp
from jax import random
from jax import device_get
import numpy as np
from functools import partial
import itertools
import time

from multi_mdn import SharedEmbedMultiMDN, train_multi_mdn, get_multiMDN_cdf_vals
from gen_gmm import GMM, train_GMM_generator, sample_GMM_gen, gmm_marg_cdf
from data import gen_mv_normal_normal_data, mvn_posterior, get_true_cdf, get_true_quantiles
from plots import plot_loss, plot_losses, plot_multi_mvn_marginals, plot_post_pairs, plot_marginal_hists
from utils import save_gen
from mdn_inv import multi_mdn_inv_marg

from utils import str2bool
from metrics import c2st, sbc, c2st_jax

import argparse

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_debug_nans", False)
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

parser = argparse.ArgumentParser()

# prelim
parser.add_argument('--folder', type=str, default="GMM_gen/normal_test3/")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--comment", type=str, default="")

# data
parser.add_argument('--d', type=int, default=10)
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--N_val', type=int, default=1000)

# MDNs set up
parser.add_argument('--mdn_lr', type=float, default=1e-3)
parser.add_argument("--post_learn_cdf", type=str2bool, default="True")
parser.add_argument('--post_mdn_K', type=int, default=1)
parser.add_argument('--post_mdn_hidden',  nargs='+', type=int, default=1*[16])
parser.add_argument('--post_mdn_epochs', type=int, default=1000)
parser.add_argument('--post_mdn_batch_size', type=int, default=1000)
parser.add_argument("--post_early_stop", type=int, default=100)

# generator architecture
parser.add_argument("--gen_K", type=int, default=1)
parser.add_argument('--gen_hidden_dims',  nargs='+', type=int, default=2*[8])

# generator training
parser.add_argument("--gen_epochs", type=int, default=100)
parser.add_argument("--gen_batch_size", type=int, default=1000)
parser.add_argument('--gen_lr', type=float, default=1e-3)
parser.add_argument('--gen_L_mc', type=int, default=400)
parser.add_argument("--gen_early_stop", type=int, default=1000)

# test
parser.add_argument("--N_test", type=int, default=1000)

args = parser.parse_args()

# set up and data
path = "experiments/" + args.folder
os.makedirs(path, exist_ok=True)

# save config
arg_dict = vars(args)
comm = arg_dict["comment"]
arg_dict.pop("comment")
with open(path+"config.txt", "w") as f:
    f.write(str(datetime.now()))
    f.write("\n\n")
    f.write(f"Comment: {comm}\n\n")
    f.write(pprint.pformat(arg_dict, width=100))
 
d = args.d
N = args.N
N_val = args.N_val

post_learn_cdf = args.post_learn_cdf
post_mdn_K = args.post_mdn_K
post_mdn_hidden = args.post_mdn_hidden
post_mdn_batch_size = args.post_mdn_batch_size
post_mdn_epochs = args.post_mdn_epochs
post_early_stop = args.post_early_stop

mdn_lr = args.mdn_lr

gen_K = args.gen_K
gen_hidden_dims = args.gen_hidden_dims
gen_epochs = args.gen_epochs
gen_batch_size = args.gen_batch_size
gen_L_mc = args.gen_L_mc
gen_lr = args.gen_lr
gen_early_stop = args.gen_early_stop


root_key = random.PRNGKey(args.seed)
keys = map(partial(random.fold_in, root_key), itertools.count()) 


L0 = jnp.sqrt(0.1) * jnp.eye(d)
L1 = jnp.sqrt(0.1) * jnp.eye(d)

# L0 = random.normal(next(keys), (d,d))
# L1 = random.normal(next(keys), (d,d))

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
    gen_hidden_dims=device_get(gen_hidden_dims),
    gen_L_mc=device_get(gen_L_mc),
    gen_K=device_get(gen_K),
    post_mdn_K=device_get(post_mdn_K),

)

print("\n----------Generating data----------\n")
x_data, θ_data = gen_mv_normal_normal_data(next(keys), 
                                            n_samples=N, 
                                            prior_mean=prior_mean,
                                            prior_L=L0, 
                                            model_L=L1)

x_mean = x_data.mean(axis=0)
x_std = x_data.std(axis=0)

post_mean, post_var = mvn_posterior(x_data, prior_mean, L0, L1)

x_data_val, θ_data_val = gen_mv_normal_normal_data(next(keys), 
                                                    n_samples=N, 
                                                    prior_mean=prior_mean,
                                                    prior_L=L0, 
                                                    model_L=L1)

post_mean_val, post_var_val = mvn_posterior(x_data_val, prior_mean, L0, L1)

print("x: ", x_data.shape, 
      "theta: ", θ_data.shape, 
      "post means: ", post_mean.shape, 
      "post var", post_var.shape)


if post_learn_cdf:
    print("\n----------Train posterior MDNs----------\n")

    post_mdn = SharedEmbedMultiMDN(var_dim=d, hidden_dims = post_mdn_hidden, K = post_mdn_K)

    post_pars_state, losses = train_multi_mdn(next(keys), 
                                                model=post_mdn, x_data=x_data, θ_data=θ_data, 
                                                lr=mdn_lr, n_epochs=post_mdn_epochs, batch_size=post_mdn_batch_size, 
                                                save_path=path+"post_mdn/",
                                                # x_val=x_data_val, theta_val=θ_data_val, early_stop=post_early_stop
                                                )
    post_pars = post_pars_state.params

    # plot_losses(losses_list, path+"post_mdn/", val_losses_list)

    test_ids = random.choice(next(keys), N, (4,))
    x_test = x_data_val[test_ids]

    plot_multi_mvn_marginals(post_mdn, post_pars, x_test, prior_mean, L0, L1, theta_range=(-5, 5), save_path = path + "post_mdn/")
    plot_loss(losses, path+"post_mdn/")

    u = get_multiMDN_cdf_vals(model=post_mdn, params=post_pars, x_data=x_data, θ_data=θ_data)
    u_val = get_multiMDN_cdf_vals(model=post_mdn, params=post_pars, x_data=x_data_val, θ_data=θ_data_val)

else:
    u = get_true_cdf(theta=θ_data, post_mean=post_mean, post_var=post_var)
    u_val = get_true_cdf(theta=θ_data_val, post_mean=post_mean_val, post_var=post_var_val)


print("\n----------Train generator----------\n")

gen = GMM(hidden_dims=gen_hidden_dims, K = gen_K, d = d)

gen_keys = map(partial(random.fold_in, next(keys)), itertools.count())

gen_state, losses  = train_GMM_generator(gen_keys,
                                            model=gen,
                                            u=u, x=x_data,
                                            learning_rate=gen_lr, 
                                            n_epochs=gen_epochs, batch_size=gen_batch_size,
                                            L_mc=gen_L_mc
                                            )

save_gen(path, gen_state.params)

plot_loss(losses, path+"gen/")

print("\n----------Test model----------\n")
# test: posterior sampling
test_locs = 4
N_test = args.N_test

test_ids = random.choice(next(keys), N_val, (test_locs,))
x_test = x_data_val[test_ids] # (test_locs, d)
# expand to match shape of v later
x_test_exp = jnp.repeat(x_test[:, None, :], axis=1, repeats=N_test)

logits, means, chol_pars = gen.apply(gen_state.params, x_test)

y_samples = sample_GMM_gen(next(keys), N_test, logits, means, chol_pars) # (test_locs, N_test, d)

print("plot y marginal hists")
plot_marginal_hists(y_samples, x_test, save_path=path + "y_mdn/", title=rf"$p(y_j \mid x)$", name="y_marg_hist.pdf")

plot_post_pairs(
    theta=y_samples,
    x_vals=x_test,
    save_dir=os.path.join(path, "y_mdn/pairplots"),
    file_prefix="bivariate_samples",
    ax_lab=rf"y",
)

# map into learned copula space

v = gmm_marg_cdf(y_samples, logits, means, chol_pars)

print("plot v marginal hists")
# should be uniform if MDNs learned correctly
plot_marginal_hists(v, x_test, save_path= path + "y_mdn/", name="marg_y_cop_hist.pdf")

# map into parameter space
if post_learn_cdf:
    print("Using learned marginal posteriors to map into parameter space")
    theta = multi_mdn_inv_marg(model = post_mdn, params=post_pars, x=x_test_exp, u=v) # (test_locs, N_test, d)
else:
    print("Using true quantile function to map into parameter space")
    theta = get_true_quantiles(u=v, x = x_test, prior_mean=prior_mean, prior_L=L0, model_L=L1)

# Ground truth posterior at x_test
mu_gt  = post_mean_val[test_ids] # (test_locs, d)  
cov_gt = jnp.repeat(post_var_val[None, ...], axis=0, repeats=test_locs) # (test_locs, d, d)

print("plot post pairs")

plot_post_pairs(
    theta=theta,
    x_vals=x_test,
    post_mean=mu_gt,
    post_cov=cov_gt,
    save_dir=os.path.join(path, "pairplots"),
    file_prefix="posterior_pairs",
)

# generate true posterior sample
eps = random.normal(next(keys), (test_locs, N_test, d)) # (test_locs, N_test, d)

L_post = jnp.linalg.cholesky(cov_gt) # (test_loc, d, d)

theta_true = mu_gt[:, None] + jnp.einsum("tnij, tnj -> tni", L_post[:, None], eps) # (test_locs, N_test, d)

plot_post_pairs(
    theta=theta_true,
    x_vals=x_test,
    post_mean=mu_gt,
    post_cov=cov_gt,
    save_dir=os.path.join(path, "pairplots"),
    file_prefix="true_posterior_pairs",
)

print("\npairplots saved\n")

# print("Performing C2ST")

t0 = time.perf_counter()
scores = [round(float(c2st_jax(keys, theta[i], theta_true[i])), 2) for i in range(test_locs)]
t1 = time.perf_counter()
print(scores, f"{t1-t0:.2f}s")

# # t0 = time.perf_counter()
# # scores = [round(float(c2st(keys, theta[i], theta_true[i])[0]), 2) for i in range(test_locs)]
# # t1 = time.perf_counter()
# # print(scores, f"{t1-t0:.2f}s")

with open(path + "metrics.txt", "w") as f:
    f.write(f"C2ST scores  {scores}")


print("performing SBC")
B = 10000
n = 100

x_samples, prior_samples = gen_mv_normal_normal_data(next(keys), 
                                                     n_samples=B, 
                                                     prior_mean=prior_mean, 
                                                     prior_L=L0, model_L=L1 )
x_samples_exp = jnp.repeat(x_samples[:, None, :], repeats = n, axis = 1 )

logits, means, chol_pars = gen.apply(gen_state.params, x_samples)
y_samples = sample_GMM_gen(next(keys), n, logits, means, chol_pars) # (B, n, d)

# map into learned copula space
v = gmm_marg_cdf(y_samples, logits, means, chol_pars)

# map into parameter space
if post_learn_cdf:
    print("Using learned marginal posteriors to map into parameter space")
    theta = multi_mdn_inv_marg(model = post_mdn, params=post_pars, x=x_samples_exp, u=v) # (B, n, d)
else:
    print("Using true quantile function to map into parameter space")
    theta = get_true_quantiles(u=v, x = x_samples, prior_mean=prior_mean, prior_L=L0, model_L=L1)

sbc_path = path + "sbc/"
os.makedirs(sbc_path, exist_ok=True)
sbc(prior_samples=prior_samples, post_samples=theta, save_path=sbc_path)

print("done!")

