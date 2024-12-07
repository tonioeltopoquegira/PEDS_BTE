import os
import orbax.checkpoint as ocp
from jax import random
import jax.numpy as jnp
from flax.training import train_state
import shutil


def init_params(model, seed, dataset_shape):

    rng = random.PRNGKey(seed)
    shape_input = dataset_shape[1:]
    dummy_batch_size = (17,)
    shape_input = dummy_batch_size + shape_input
    params = model.init(rng, jnp.ones(shape_input))['params']

    return params

def update_params(path, model, seed, dataset_shape):

    if os.path.exists(f"weights/{path}") and os.listdir(f"weights/{path}") and False:
        print("Restoring present params checkpoint")
        # Try restoring the checkpoint
        checkpointer = ocp.PyTreeCheckpointer()
        options = ocp.CheckpointManagerOptions(max_to_keep=1, create=False)
        save_path_old = os.path.abspath(f"weights/{path}")

        mngr_restored = ocp.CheckpointManager(save_path_old, checkpointer, options)

        try:
            restored_ckpt = mngr_restored.restore(mngr_restored.latest_step())
            # Parameters from checkpoint
            final_params = restored_ckpt["state"]["params"]
        except FileNotFoundError:
            print(f"Warning: No valid checkpoint found in {save_path_old}. Initializing new parameters.")
            final_params = init_params(model, seed, dataset_shape)
            mngr_restored = None  # No manager if restore failed
        
    else:
        print("Initialize params from scratch")
        # Otherwise, create a new Checkpoint Manager for saving new parameters
        save_path = os.path.abspath(f"weights/{path}")

        checkpointer_new = ocp.PyTreeCheckpointer()
        options_new = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
        mngr_new = ocp.CheckpointManager(save_path, checkpointer_new, options_new)

        final_params = init_params(model, seed, dataset_shape)

    # Return the relevant manager (restored or new)
    return final_params, mngr_restored if 'mngr_restored' in locals() else mngr_new

