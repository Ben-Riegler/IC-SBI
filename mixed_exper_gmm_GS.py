import os
from datetime import datetime
import pprint
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
import itertools
import time

from mixed_mdn import (
    MixedMarginalMDN,
    train_mixed_mdn,
    get_mixed_cdf_vals,
    mixed_mdn_inv_marg,
    plot_abc_vs_mdn
)
from gen_gmm_GS import (
    GMM,
    train_GMM_generator,
    sample_GMM_gen,
    gmm_marg_cdf,
    chol_pars_to_L,
)
from data import gen_mixed_prior_binomial_normal_data
from plots import plot_loss, plot_post_pairs, plot_marginal_hists
from utils import save_gen, str2bool
from metrics import c2st_jax, sbc

import argparse

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_debug_nans", False)
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

parser = argparse.ArgumentParser()

# prelim
parser.add_argument("--folder", type=str, default="GMM_gen/mixed_test11/")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--comment", type=str, default="Dependent mixed prior Gumbel Softmax")

# data
parser.add_argument("--d", type=int, default=3)
parser.add_argument("--N", type=int, default=100000)
parser.add_argument("--N_val", type=int, default=1000)
parser.add_argument("--dependence", type=str2bool, default="True")

# marginal MDN
parser.add_argument("--mdn_lr", type=float, default=5e-4)
parser.add_argument("--post_mdn_K", type=int, default=1)
parser.add_argument("--post_mdn_hidden", nargs="+", type=int, default=2*[64])
parser.add_argument("--post_mdn_epochs", type=int, default=1000)
parser.add_argument("--post_mdn_batch_size", type=int, default=100000)

# generator GMM
parser.add_argument("--gen_K", type=int, default=2)
parser.add_argument("--gen_hidden_dims", nargs="+", type=int, default=2*[32])
parser.add_argument("--gen_epochs", type=int, default=800)
parser.add_argument("--gen_batch_size", type=int, default=50000)
parser.add_argument("--gen_lr", type=float, default=1e-3)
parser.add_argument("--gen_L_mc", type=int, default=50)

# test
parser.add_argument("--N_test", type=int, default=1000)
parser.add_argument("--test_locs", type=int, default=20)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
path = "experiments/" + args.folder
os.makedirs(path, exist_ok=False)

d = args.d  # always 3 for this DGP

arg_dict = vars(args)
comm = arg_dict.pop("comment")
with open(path + "config.txt", "w") as f:
    f.write(str(datetime.now()) + "\n\n")
    f.write(f"Comment: {comm}\n\n")
    f.write(pprint.pformat(arg_dict, width=100))

root_key = random.PRNGKey(args.seed)
keys = map(partial(random.fold_in, root_key), itertools.count())

# ---------------------------------------------------------------------------
# 1. Generate data
# ---------------------------------------------------------------------------
print("\n---------- Generating data ----------\n")

x_data, theta_data = gen_mixed_prior_binomial_normal_data(next(keys), args.N, args.dependence)
x_data_val, theta_data_val = gen_mixed_prior_binomial_normal_data(next(keys), args.N_val, args.dependence)

print("x:", x_data.shape, "theta:", theta_data.shape)

# ---------------------------------------------------------------------------
# 2. Train marginal posterior model
# ---------------------------------------------------------------------------
print("\n---------- Train marginal posterior MDN ----------\n")

post_mdn = MixedMarginalMDN(
    hidden_dims=args.post_mdn_hidden,
    K=args.post_mdn_K,
)

post_state, post_losses = train_mixed_mdn(
    next(keys),
    model=post_mdn,
    x_data=x_data,
    theta_data=theta_data,
    lr=args.mdn_lr,
    n_epochs=args.post_mdn_epochs,
    batch_size=args.post_mdn_batch_size,
    save_path=path + "post_mdn/",
)
post_params = post_state.params

plot_loss(post_losses, path + "post_mdn/")

plot_abc_vs_mdn(
    model=post_mdn,
    params=post_params,
    key=next(keys),
    x_data=x_data_val,   # use validation data to pick test points
    n_obs=4,
    n_accept=5000,
    save_path=path + "post_mdn/",
)

# Compute PIT values
u = get_mixed_cdf_vals(post_mdn, post_params, x_data, theta_data)
u_val = get_mixed_cdf_vals(post_mdn, post_params, x_data_val, theta_data_val)

# Quick PIT histogram check
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
names = ["t1 (Gauss)", "t2 (Gamma)", "t3 (Beta)"]
for j in range(3):
    axes[j].hist(np.array(u_val[:, j]), bins=30, density=True)
    axes[j].axhline(1.0, color="r", ls="--")
    axes[j].set_title(f"PIT {names[j]}")
plt.tight_layout()
plt.savefig(path + "post_mdn/pit_check.pdf")
plt.close()

# ---------------------------------------------------------------------------
# 3. Train copula generator
# ---------------------------------------------------------------------------
print("\n---------- Train generator ----------\n")

gen = GMM(hidden_dims=args.gen_hidden_dims, K=args.gen_K, d=d)

gen_keys = map(partial(random.fold_in, next(keys)), itertools.count())

gen_state, gen_losses = train_GMM_generator(
    gen_keys,
    model=gen,
    u=u,
    x=x_data,
    learning_rate=args.gen_lr,
    n_epochs=args.gen_epochs,
    batch_size=args.gen_batch_size,
    L_mc=args.gen_L_mc,
)

save_gen(path, gen_state.params)
plot_loss(gen_losses, path + "gen/")

# ---------------------------------------------------------------------------
# 4. Test: posterior sampling
# ---------------------------------------------------------------------------
print("\n---------- Test model ----------\n")

test_locs = args.test_locs
N_test = args.N_test

test_ids = random.choice(next(keys), args.N_val, (test_locs,))
x_test = x_data_val[test_ids]  # (test_locs, 1)
x_test_exp = jnp.repeat(x_test[:, None, :], axis=1, repeats=N_test)  # (test_locs, N_test, 1)

# sample from generator
logits, means, chol_pars = gen.apply(gen_state.params, x_test)
y_samples = sample_GMM_gen(next(keys), N_test, logits, means, chol_pars)  # (test_locs, N_test, d)

print("plot y marginal hists")
plot_marginal_hists(
    y_samples, x_test,
    save_path=path + "y_mdn/",
    title=r"$p(y_j \mid x)$",
    name="y_marg_hist.pdf",
)

# map into copula space
v = gmm_marg_cdf(y_samples, logits, means, chol_pars_to_L(chol_pars, d))

print("plot v marginal hists (should be uniform)")
plot_marginal_hists(
    v, x_test,
    save_path=path + "y_mdn/",
    name="marg_y_cop_hist.pdf",
    bins=10,
)

# map into parameter space via learned inverse marginals
print("Using learned marginal posteriors to map into parameter space")
theta = mixed_mdn_inv_marg(
    model=post_mdn,
    params=post_params,
    x=x_test_exp,
    u=v,
)  # (test_locs, N_test, 3)

print("plot posterior pair plots")
plot_post_pairs(
    theta=theta,
    x_vals=x_test,
    save_dir=os.path.join(path, "pairplots"),
    file_prefix="posterior_pairs",
    ax_lab=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"],
)

# In mixed_exper_gmm_GS.py, replace the C2ST section with:
from abc_reference import abc_rejection_multi

theta_abc = abc_rejection_multi(
    next(keys), x_test.squeeze(), 
    n_accept=N_test, eps=0.1
)  # (test_locs, N_test, 3)

print("Performing C2ST")

t0 = time.perf_counter()
scores = [round(float(c2st_jax(keys, theta[i], theta_abc[i])), 2) for i in range(test_locs)]
t1 = time.perf_counter()
print(scores, jnp.stack(scores).mean(),f"{t1-t0:.2f}s")

with open(path + "metrics.txt", "w") as f:
    f.write(f"C2ST scores  {scores}")


print("\n---------- SBC ----------\n")

B_sbc = 10000
n_sbc = 100

x_sbc, prior_sbc = gen_mixed_prior_binomial_normal_data(next(keys), B_sbc)
x_sbc_exp = jnp.repeat(x_sbc[:, None, :], repeats=n_sbc, axis=1)

logits_sbc, means_sbc, chol_pars_sbc = gen.apply(gen_state.params, x_sbc)
y_sbc = sample_GMM_gen(next(keys), n_sbc, logits_sbc, means_sbc, chol_pars_sbc)

v_sbc = gmm_marg_cdf(y_sbc, logits_sbc, means_sbc, chol_pars_to_L(chol_pars_sbc, d))

theta_sbc = mixed_mdn_inv_marg(
    model=post_mdn,
    params=post_params,
    x=x_sbc_exp,
    u=v_sbc,
)

sbc_path = path + "sbc/"
os.makedirs(sbc_path, exist_ok=True)
t0 = time.perf_counter()
sbc(prior_samples=prior_sbc.astype(jnp.float32), post_samples=theta_sbc.astype(jnp.float32), save_path=sbc_path)
t1 = time.perf_counter()
print(f"This took {t1-t0:.2f}s")

print("done!")
