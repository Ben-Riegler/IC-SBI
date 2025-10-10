import os
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import orbax_utils


# =========================
#   MDN: SAVE / LOAD
# =========================

def save_MDN(path: str, dim: int, params):
    """
    Save MDN params using Orbax, keeping structure {'params': ...} for compatibility.
    Creates .../ckpt_mdn_{dim}/mdn_0 as the checkpoint folder.
    """
    base = os.path.abspath(os.path.join(path, f"ckpt_mdn_{dim}"))
    ckpt_dir = os.path.join(base, "mdn_0")   # one checkpoint named "mdn_0"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpointer = ocp.PyTreeCheckpointer()
    target = {'params': params}
    save_args = orbax_utils.save_args_from_target(target)
    checkpointer.save(ckpt_dir, target, save_args=save_args, force=True)
    print("Saved MDN params to", ckpt_dir)


def load_MDN(ckpt_dir: str, model_def, rng_key, hidden_dims, K, x_dim):
    """
    Load a trained MDN (Orbax). `ckpt_dir` should point to the checkpoint folder,
    e.g. .../ckpt_mdn_{dim}/mdn_0
    Returns (model, {'params': ...}) for compatibility with your code.
    """
    # Init a dummy model to get the expected param structure/dtypes
    model = model_def(hidden_dims=hidden_dims, K=K)
    dummy_x = jnp.zeros((1, x_dim))
    init_vars = model.init(rng_key, dummy_x)

    target = {'params': init_vars['params']}
    restore_args = orbax_utils.restore_args_from_target(target)

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_dir, item=target, restore_args=restore_args)
    return model, restored


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
    target = {'params': params}
    save_args = orbax_utils.save_args_from_target(target)
    checkpointer.save(ckpt_dir, target, save_args=save_args, force=True)
    print("Saved generator params to", ckpt_dir)



# import jax.numpy as jnp
# from flax.training import checkpoints
# import os

# def save_MDN(path, dim, params):
#     ckpt_dir = os.path.join(path, f"ckpt_mdn_{dim}")
#     ckpt_dir = os.path.abspath(ckpt_dir)
#     os.makedirs(ckpt_dir, exist_ok=True)
#     checkpoints.save_checkpoint(
#         ckpt_dir,
#         target=params,
#         step=0,
#         prefix="mdn_",
#         overwrite=True
#     )
#     print("Saved MDN params to", ckpt_dir)

# def load_MDN(ckpt_dir, model_def, rng_key, hidden_dims, K, x_dim):
#     """
#     Load a trained MDN model from checkpoint in a notebook.

#     Args:
#       ckpt_dir: directory containing checkpoints
#       model_def: MDN class definition
#       rng_key: PRNGKey for parameter init
#       hidden_dims: same architecture used during training

#     Returns:
#       model: Flax module instance
#       params: loaded parameters
#     """
#     # Initialize model and dummy params
#     model = model_def(hidden_dims= hidden_dims, K = K)
#     dummy_x = jnp.zeros((1, x_dim))
#     init_vars = model.init(rng_key, dummy_x)

#     # Restore
#     restored = checkpoints.restore_checkpoint(
#         ckpt_dir,
#         target={'params': init_vars['params']},
#         prefix='mdn_'
#     )
#     return model, restored


# def save_gen(path, params):
    
#     ckpt_dir = os.path.join(path, f"gen/ckpt_gen")
#     ckpt_dir = os.path.abspath(ckpt_dir)
#     os.makedirs(ckpt_dir, exist_ok=True)
#     checkpoints.save_checkpoint(
#         ckpt_dir,
#         target=params,
#         step=0,
#         prefix="gen_",
#         overwrite=True
#     )