import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import os
import absl.logging
absl.logging.set_verbosity('error')


def hardtanh(x):
    """Hard tanh activation: max(-1, min(1, x))."""
    return jnp.clip(x, 1e-16, 160.0)


def choose_activation(activation, num_layers):
    """Choose activation function for each layer."""
    activation_functions = []
    if activation == "relu":
        activation_functions = [nnx.relu] * num_layers
    elif activation == "hardtanh":
        activation_functions = [hardtanh] * num_layers
    elif activation == "mixed":
        activation_functions = [nnx.relu] * (num_layers-1) 
        activation_functions.append(hardtanh)
    elif activation == "sigmoid":
         activation_functions = [nnx.sigmoid] * num_layers
    
    return activation_functions


class mlp(nnx.Module):
    """Multi-layer perceptron model using Flax NNX."""

    def __init__(self, layer_sizes: list, activation: str, initialization: str, rngs: nnx.Rngs):
        super().__init__()

        if initialization == "he":
            dense_init = nnx.initializers.kaiming_normal()
        elif initialization == "xavier":
            dense_init = nnx.initializers.xavier_normal()
        
        bias_init = nnx.initializers.constant(0.0)
            
        self.layers = [
            nnx.Linear(i, o, kernel_init=dense_init, bias_init=bias_init, rngs=rngs)
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        
        self.activations = choose_activation(activation, len(self.layers))

    def __call__(self, x, training=False):
        for en, (layer, activation_fn) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            x = activation_fn(x)  # Apply the activation function for this layer
    
        return x


def filter_dropout(state):
    """Filter out dropout states that can cause issues during loading."""
    if 'generator' in state:
        generator_state = state['generator']
        if 'dropout' in generator_state:
            del generator_state['dropout']
    
    if 'dropout' in state:
        del state['dropout']
    return state


def load_model_parameters(model, checkpoint_path):
    """
    Load model parameters from a checkpoint using the same approach as initialize_or_restore_params.
    
    Args:
        model: The initialized model (mlp instance)
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        model: Model with loaded parameters
    """
    try:
        # Create checkpointer
        checkpointer = ocp.StandardCheckpointer()
        
        # Split the model to get components (same as working code)
        graphdef, rng_state, abstract_state = nnx.split(model, nnx.RngState, ...)
        
        # Strip 'key' from abstract_state if present (same as working code)
        if 'key' in abstract_state:
            print("Stripping 'key' from abstract_state before restore")
            del abstract_state['key']
        
        # Restore the state from checkpoint
        ckpt_path = os.path.abspath(checkpoint_path)
        state_restored = checkpointer.restore(ckpt_path, abstract_state)
        
        # Remove any 'key' entries from restored state (same as working code)
        if 'key' in state_restored:
            print("Dropping 'key' from restored state")
            del state_restored['key']
        
        if isinstance(state_restored, dict) and 'key' in state_restored:
            print("Dropping 'key' from restored state")
            del state_restored['key']
        
        # Filter out dropout states
        state_restored = filter_dropout(state_restored)
        
        # Merge back into model (same as working code)
        model = nnx.merge(graphdef, rng_state, state_restored)
        
        print(f"Successfully loaded parameters from {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"Failed to load parameters from {checkpoint_path}: {e}")
        raise e


def load_data(data_path):
    """
    Load high fidelity data from npz file.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        pores: Input features (pore configurations)
        kappas: Target values (thermal conductivity)
    """
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


