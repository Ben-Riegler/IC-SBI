import jax.numpy as jnp
from flax.training import checkpoints
import os

def save_MDN(path, dim, params):
    ckpt_dir = os.path.join(path, f"ckpt_mdn_{dim}")
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir,
        target=params,
        step=0,
        prefix="mdn_",
        overwrite=True
    )
    print("Saved MDN params to", ckpt_dir)

def load_MDN(ckpt_dir, model_def, rng_key, hidden_dims, K, x_dim):
    """
    Load a trained MDN model from checkpoint in a notebook.

    Args:
      ckpt_dir: directory containing checkpoints
      model_def: MDN class definition
      rng_key: PRNGKey for parameter init
      hidden_dims: same architecture used during training

    Returns:
      model: Flax module instance
      params: loaded parameters
    """
    # Initialize model and dummy params
    model = model_def(hidden_dims= hidden_dims, K = K)
    dummy_x = jnp.zeros((1, x_dim))
    init_vars = model.init(rng_key, dummy_x)

    # Restore
    restored = checkpoints.restore_checkpoint(
        ckpt_dir,
        target={'params': init_vars['params']},
        prefix='mdn_'
    )
    return model, restored


def save_gen(path, params):
    ckpt_dir = os.path.join(path, f"gen/ckpt_gen")
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir,
        target=params,
        step=0,
        prefix="gen_",
        overwrite=True
    )