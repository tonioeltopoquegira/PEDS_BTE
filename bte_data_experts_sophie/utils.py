import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import os
import absl.logging
absl.logging.set_verbosity('error')

def load_model_parameters(model, checkpoint_path):
    
    try:
        # Create checkpointer
        checkpointer = ocp.StandardCheckpointer()
        
        # Split the model to get components
        graphdef, rng_state, state = nnx.split(model, nnx.RngState, ...)
        
        # Create abstract state for restoration
        abstract_state = nnx.State.from_flat_path(state.flat_paths())
        
        # Restore the state from checkpoint
        state_restored = checkpointer.restore(checkpoint_path, abstract_state)
        
        # Filter out any dropout states if present
        def filter_dropout(state):
            if hasattr(state, 'dropout'):
                delattr(state, 'dropout')
            return state
        
        state_restored = filter_dropout(state_restored)
        
        # Merge back into model
        model = nnx.merge(graphdef, rng_state, state_restored)
        
        print(f"Successfully loaded parameters from {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"Failed to load parameters from {checkpoint_path}: {e}")
        raise e


def load_data(data_path):
    
    try:
        data = jnp.load(data_path, allow_pickle=True)
        pores = jnp.asarray(data['pores'], dtype=jnp.float32)
        kappas = jnp.asarray(data['kappas'], dtype=jnp.float32)
        
        print(f"Loaded data: {pores.shape[0]} samples")
        print(f"Input shape: {pores.shape}")
        print(f"Output shape: {kappas.shape}")
        
        return pores, kappas
        
    except Exception as e:
        print(f"Failed to load data from {data_path}: {e}")
        raise e