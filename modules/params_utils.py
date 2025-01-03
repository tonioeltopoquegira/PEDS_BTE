import os
import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path
from datetime import datetime


# Function for initializing or restoring model parameters
def initialize_or_restore_params(generator, model_name, base_dir="weights"):
    """
    Initialize or restore model parameters based on the existence of a checkpoint.

    Parameters:
        generator: Callable
            The model generator function.
        model_name: str
            Name of the model for checkpointing.
        base_dir: str
            Base directory for storing weights.

    Returns:
        model: nnx.Model
            The initialized or restored model.
    """
    # Define paths
    ckpt_dir = Path(base_dir) / model_name
    
    # Ensure checkpoint directory exists
    if not ckpt_dir.exists():
        print(f"Checkpoint directory does not exist. Creating new directory at {ckpt_dir}.")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create the checkpointer
    checkpointer = ocp.StandardCheckpointer()

    # Generate the abstract model
    graphdef, abstract_state = nnx.split(generator)

    try:
        last_checkpoint = max(
            (d for d in ckpt_dir.iterdir() if d.is_dir()),  # Consider all directories
            key=lambda x: x.name.split('_')[-2] + '_' + x.name.split('_')[-1],  # Use the last two parts as timestamp
        )
        print(f"Found last checkpoint: {last_checkpoint}")
        
        # Attempt to restore the state
        try:
            state_restored = checkpointer.restore(last_checkpoint , abstract_state)
            print(f"Successfully restored state from {last_checkpoint}.")
        except Exception as e:
            print(f"Restoration failed with error: {e}. Initializing new parameters.")
            state_restored = abstract_state  # Initialize new state

    except ValueError:
        # If no checkpoint is found
        print("No checkpoints found. Initializing new parameters.")
        state_restored = abstract_state  # Initialize new state


    # Merge graph definition with restored or initialized state
    model = nnx.merge(graphdef, state_restored)
    return model, checkpointer, ckpt_dir

def save_params(generator, checkpointer, epoch=None):

    _, state = nnx.split(generator)
    #nnx.display(state)
    base_dir = os.path.abspath('weights/base_peds')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')  # e.g., "20241213_123456"
    if epoch is None:
        timestamp = f'final_' + timestamp
    else:
        timestamp = f'epoch{epoch}_' + timestamp
    
    new_checkpoint_dir = os.path.join(base_dir, timestamp)

    #os.makedirs(new_checkpoint_dir, exist_ok=False)  # Ensure the directory does not already exist
    checkpointer.save(new_checkpoint_dir, state)