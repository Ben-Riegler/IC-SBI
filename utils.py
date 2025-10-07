import jax.numpy as jnp
from flax.training import checkpoints

def load_model(ckpt_dir, model_def, rng_key, hidden_dims= 1 * [8], K = 1, d = 1):
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
    dummy_x = jnp.zeros((d,1))
    init_vars = model.init(rng_key, dummy_x)

    # Restore
    restored = checkpoints.restore_checkpoint(
        ckpt_dir,
        target={'params': init_vars['params']},
        prefix='mdn_'
    )
    return model, restored