import os
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import jax
import flax.linen as nn

class standardize(nn.Module):
    mean: jnp.ndarray # (1, d)
    std: jnp.ndarray # (1, d)

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray # (N, d)
                 ):
        
        d = x.shape[-1]
        assert d == self.mean.shape[-1]
        assert d == self.std.shape[-1]

        return (x - self.mean) / ( self.std)

def standardize_cols(X, mean=None, std=None):
    """
    Standardize each column of X to have mean 0 and std 1.
    Returns the standardized X, along with the column means and stds.
    """
    if mean is None:
        mean = jnp.mean(X, axis=0, keepdims=True) # (1, d)
    if std is None:
        std = jnp.std(X, axis=0, keepdims=True) + 1e-8 # (1, d)
    X_std = (X - mean) / std
    return X_std, mean, std


def str2bool(flag):
    if flag =="True":
        return True
    elif flag =="False":
        return False
    else:
        raise ValueError("Need True or False")


def save_MDN(ckpt_dir: str, dim,  params) -> None:
    """
    Save a Flax params pytree to `ckpt_dir` using Orbax.
    """
    ckpt_dir = os.path.abspath(ckpt_dir + f"ckpt_{dim}/")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()

    # Save the params pytree (no wrapper)
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(ckpt_dir, params, save_args=save_args, force=True)
    print(f"Saved MDN for dimension {dim} to {ckpt_dir}")


def load_MDN(ckpt_dir: str, model_def, rng_key, hidden_dims, K, x_dim):
    """
    Build and init the model, then restore params from `ckpt_dir`.
    Returns (model, restored_params).
    """

    model = model_def(hidden_dims=hidden_dims, K=K)

    # Initialize to get a like-shaped params pytree
    # dummy_x = jnp.zeros((1, x_dim))
    # init_vars = model.init(rng_key, dummy_x)          # {'params': ...}
    # params_like = init_vars["params"]                  # <-- match what you saved

    # print(params_like)

    # # Restore directly into the like-shaped params pytree, with restore_args
    # checkpointer = ocp.PyTreeCheckpointer()
    # restore_args = orbax_utils.restore_args_from_target(params_like)
    # restored_params = checkpointer.restore(
    #     ckpt_dir, item=params_like, restore_args=restore_args
    # )
    # # 

    # # Restore directly into the like-shaped pytree (no restore_args needed)
    # checkpointer = ocp.PyTreeCheckpointer()
    # restored_params = checkpointer.restore(ckpt_dir, 
    #                                        # item=init_vars
    #           
    
     # Build a like-shaped params tree ON CPU so restore_args has CPU sharding.
    cpu0 = jax.devices("cpu")[0]
    dummy_x = jnp.zeros((1, x_dim))

    with jax.default_device(cpu0):
        init_vars = model.init(rng_key, dummy_x)   # {'params': ...}

    # Make sure every leaf is placed/sharded on CPU explicitly.
    params_like = jax.tree.map(lambda a: jax.device_put(a, cpu0), init_vars)

    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = orbax_utils.restore_args_from_target(params_like)

    # Restore into params_like (NOT init_vars).
    restored_params = checkpointer.restore(
        ckpt_dir,
        item=params_like,
        restore_args=restore_args,
    )

                           

    return model, restored_params


def save_multiMDN(path: str, params):
    """
    Save multiMDN params using Orbax, keeping structure {'params': ...} for compatibility.
    Creates .../ckpt_multi_mdn/multi_mdn_0 as the checkpoint folder.
    """
    base = os.path.abspath(os.path.join(path, f"ckpt_multi_mdn"))
    ckpt_dir = os.path.join(base, "multi_mdn_0")   
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()
    target = {'params': params}
    save_args = orbax_utils.save_args_from_target(target)
    checkpointer.save(ckpt_dir, target, save_args=save_args, force=True)
    print("Saved multiMDN params to", ckpt_dir)


# =========================
#   GENERATOR: SAVE ONLY
# =========================

def save_gen(path: str, params):
    """
    Save generator params using Orbax, keeping structure {'params': ...}.
    Creates .../gen/ckpt_gen/gen_0 as the checkpoint folder.
    """
    base = os.path.abspath(os.path.join(path, "gen", "ckpt_gen"))
    ckpt_dir = os.path.join(base, "gen_0")

    os.makedirs(ckpt_dir, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()

    # Save the params pytree (no wrapper)
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(ckpt_dir, params, save_args=save_args, force=True)
    print("Saved generator params to", ckpt_dir)


def load_gen(ckpt_dir: str, model_def, rng_key, emb_dim, hidden_dims, z_dim, x_dim, out_dim):
    """
    Build and init the model, then restore params from `ckpt_dir`.
    Returns (model, restored_params).
    """

    model = model_def(emb_dim=emb_dim, hidden_dims=hidden_dims, out_dim=out_dim)

    # Initialize to get a like-shaped params pytree
    dummy_z = jnp.zeros((1, z_dim))
    dummy_x = jnp.zeros((1, x_dim))


     # Build a like-shaped params tree ON CPU so restore_args has CPU sharding.
    cpu0 = jax.devices("cpu")[0]

    with jax.default_device(cpu0):
        init_vars = model.init(rng_key, dummy_z, dummy_x)          # {'params': ...}

    # Make sure every leaf is placed/sharded on CPU explicitly.
    params_like = jax.tree.map(lambda a: jax.device_put(a, cpu0), init_vars)

    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = orbax_utils.restore_args_from_target(params_like)

    # Restore into params_like (NOT init_vars).
    restored_params = checkpointer.restore(
        ckpt_dir,
        item=params_like,
        restore_args=restore_args,
    )


    return model, restored_params


