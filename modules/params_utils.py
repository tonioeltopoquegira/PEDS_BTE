import os
import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path
from datetime import datetime
import shutil
import jax
import jax.numpy as jnp
import jax.random as random

import absl.logging
absl.logging.set_verbosity('error')


from models.ensembles import ensemble

def get_shapes(params):
    return jax.tree_util.tree_map(lambda x: x.shape if isinstance(x, jnp.ndarray) else None, params)

def filter_dropout(state):
    if 'generator' in state:
        generator_state = state['generator']
        if 'dropout' in generator_state:
            del generator_state['dropout']
    
    if 'dropout' in state:
        del state['dropout']
    return state

import os
from datetime import datetime
from pathlib import Path
import jax.random as random
from flax import nnx
import orbax.checkpoint as ocp

# assume `ensemble` and `filter_dropout` are defined elsewhere

def initialize_or_restore_params(generator, retrain, model_name, rank, base_dir, seed=None):
    """
    Initialize or restore model parameters based on the existence of a checkpoint.
    """

    if isinstance(generator, ensemble):
        return initialize_or_restore_ensemble(generator, retrain, model_name, rank, base_dir, seed)
   
    if seed is not None:
        _ = random.PRNGKey(seed)

    ckpt_dir = Path(base_dir) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    graphdef, rng_state, abstract_state = nnx.split(generator, nnx.RngState, ...)
    if 'key' in abstract_state:
        if rank == 0:
            print("Stripping 'key' from abstract_state before restore")
        del abstract_state['key']

    try:
        last_ckpt = max(
            (d for d in ckpt_dir.iterdir() if d.is_dir()),
            key=lambda d: d.name.rsplit('_', 2)[-2] + "_" + d.name.rsplit('_', 2)[-1],
        )
        if rank == 0:
            print(f"Found last checkpoint: {last_ckpt}")

        if not retrain:
            try:
                ckpt_path = os.path.abspath(last_ckpt)
                
                state_restored = checkpointer.restore(ckpt_path, abstract_state)

                if 'key' in state_restored:
                    if rank == 0:
                        print("Dropping 'key' from restored state")
                    del state_restored['key']

                # === DROP any stray 'key' subtree ===
                if isinstance(state_restored, dict) and 'key' in state_restored:
                    if rank == 0:
                        print("Dropping 'key' from restored state")
                    del state_restored['key']
            

                state_restored = filter_dropout(state_restored)
                if rank == 0:
                    print(f"Successfully restored state from {last_ckpt}")
            except Exception as e:
                if rank == 0:
                    print(f"Restoration failed ({e}); initializing new parameters")
                state_restored = abstract_state
        else:
            if rank == 0:
                print("Retrain mode: skipping restore, initializing new parameters")
            state_restored = abstract_state

    except ValueError:
        if rank == 0:
            print("No checkpoints found; initializing new parameters")
        state_restored = abstract_state

    model = nnx.merge(graphdef, rng_state, state_restored)
    return model, checkpointer


def save_params(exp_name, model_name, generator, checkpointer, epoch=None):
    """
    Save model parameters, dropping any 'key' subtree so it never gets serialized.
    """

    if isinstance(generator, ensemble):
        save_params_ensemble(exp_name, model_name, generator, checkpointer, epoch)
        return

    graphdef, rng_state, state = nnx.split(generator, nnx.RngState, ...)

    if 'key' in state:
        print("Stripping 'key' from state before save")
        del state['key']

    state = filter_dropout(state)

    base = os.path.abspath(f"experiments/{exp_name}/weights/{model_name}")
    tag = f"epoch{epoch}_" if epoch is not None else "final_"
    tag += datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(base, tag)

    checkpointer.save(save_dir, state)


   
def initialize_or_restore_ensemble(ensemble, retrain, model_name, rank, base_dir, seed):
    n_model = ensemble.n_models
    checkpointers = []

    for i in range(n_model):
        model_name_i = f"{model_name}/model_{i}"  # only model_name changes
        ensemble.models[i], checkpointer = initialize_or_restore_params(
            ensemble.models[i],
            retrain,
            model_name_i,
            rank,
            base_dir=base_dir,  # base_dir stays same!
            seed=seed + i
        )
        checkpointers.append(checkpointer)

    return ensemble, checkpointers



def save_params_ensemble(exp_name, model_name, ensemble, checkpointers, epoch=None):
    for i in range(ensemble.n_models):
        model_name_i = f"{model_name}/model_{i}"  # only name changes
        save_params(exp_name, model_name_i, ensemble.models[i], checkpointers[i], epoch)
