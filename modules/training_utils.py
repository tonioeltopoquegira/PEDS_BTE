import os
import sys
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import optax
from flax import nnx
import pandas as pd
import seaborn as sns
import json

from mpi4py import MPI
from models.ensembles import ensemble
from modules.params_utils import save_params

cmap = plt.cm.viridis

# Define local data loader
def data_loader(*arrays, batch_size):
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(array[batch_indices] for array in arrays)



def clip_gradients(grads, clip_value=1.0):
    return jax.tree_util.tree_map(lambda g: jnp.clip(g, -clip_value, clip_value), grads)


def plot_learning_curves(exp_name, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, model_name, epoch, learn_rate_max, learn_rate_min):
    """Plot the learning curve and validation percentage losses."""
    epochs = jnp.arange(epoch)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot: Training and validation losses
    axs[0].plot(epochs, epoch_losses[:epoch], 'bo-', label='Training Loss')
    axs[0].plot(epochs, valid_losses[:epoch], 'ro-', label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(f'Learning Curve - Rates [{learn_rate_min}, {learn_rate_max}]')
    axs[0].legend()
    axs[0].grid()

    # Second subplot: Validation percentage loss
    axs[1].plot(epochs, valid_perc_losses[:epoch], 'go-', label='Validation % Loss')
    axs[1].axhline(5.0, color='gray', linestyle='--', label='5.00% Line')  # Add the 5.00% line
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Validation % Loss')
    axs[1].set_title('Validation % Loss Evolution')
    axs[1].legend()
    axs[1].grid()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{exp_name}/figures/learning_curve_{model_name}.png")
    plt.close()

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot: Training and validation losses
    axs[0].plot(epochs, np.cumsum(epoch_times[:epoch])/epochs[:epoch], 'bo-', label='Epoch Speed')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Time (s)')
    axs[0].set_title(f'Time to complete each epoch')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(np.cumsum(epoch_times[:epoch]), valid_perc_losses[:epoch], 'ro-', label='Mean Percentual error')
    axs[0].set_xlabel('Epoch (#)')
    axs[0].set_ylabel('Time (s)')
    axs[0].set_title(f'Average Time per epoch')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Mean Percentual Error (%)')
    axs[1].set_title(f'Mean Percentual Error over Time')
    axs[1].legend()
    axs[1].grid()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{exp_name}/figures/learning_curve_{model_name}.png")
    plt.close()


def update_and_check_grads(grads, grads_new):
    """Update gradients and check for NaN or infinity values."""
    # Accumulate gradients
    if grads is None:
        grads = grads_new
    else:
        grads = jax.tree_util.tree_map(lambda g, gn: g + gn, grads, grads_new)

    # Check for NaN or infinity gradients
    has_nan_or_inf = jax.tree_util.tree_reduce(
        lambda acc, leaf: acc or jnp.isinf(leaf).any() or jnp.isnan(leaf).any(), 
        grads, 
        initializer=False
    )

    if has_nan_or_inf:
        print("The gradients contain infinity or NaN values.")

    # Check for zero gradients
    has_zero_grads = jax.tree_util.tree_reduce(
        lambda acc, leaf: acc or (leaf == 0).all(), 
        grads, 
        initializer=False
    )

    if has_zero_grads:
        print("The gradients are all zero.")

    if has_nan_or_inf or has_zero_grads:
        raise ValueError("Gradient check failed.")

    return grads


#@nnx.jit
def accumulate_gradients(total_grads, new_grads):
        if total_grads is None:
            return new_grads
        return jax.tree_util.tree_map(lambda x, y: x + y, total_grads, new_grads)


#@nnx.jit
def hardtanh(x):
    """Hard tanh activation: max(-1, min(1, x))."""
    return jnp.clip(x, 1e-16, 160.0)


def choose_schedule(rank, schedule, learn_rate_min, learn_rate_max, epochs, exp_name, n_past_epochs):
    
    if schedule == "cosine-decay":
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learn_rate_max,  # Maximum learning rate
            decay_steps=epochs,   # Number of epochs to decay over
            alpha=learn_rate_min / learn_rate_max  # Minimum learning rate as a fraction of max
        )

    if schedule == "cosine-cycles":

        cycle = 25  # Adjust as needed
        peak_value = learn_rate_max  # Max LR
        min_value = learn_rate_min  # Min LR
        lr_schedule = CustomCosineCycleSchedule(cycle, peak_value, min_value)
    
    if schedule == "exponential":
        lr_schedule = optax.exponential_decay(
            init_value=learn_rate_max,
            transition_steps=1000,
            decay_rate=learn_rate_min / learn_rate_max
        )

    if schedule == "constant":
        lr_schedule = learn_rate_min
        return lr_schedule
    
    
    save_dir = f"experiments/{exp_name}/figures"
    os.makedirs(save_dir, exist_ok=True)

    if rank == 0 and n_past_epochs == 0:

        steps = np.arange(100)
        lr_values = np.array([lr_schedule(step) for step in steps])
        # Plotting the learning rate schedule
        plt.plot(steps, lr_values, label=schedule)
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.xlim(0, 100)
        plt.yscale('log')
        plt.title(f'Learning Rate Schedule ({schedule})')
        plt.legend()

        # Save the plot
        plt.savefig(f"{save_dir}/learn_schedule_{schedule}.png")
        plt.close()
    
    return lr_schedule


class CustomCosineCycleSchedule(optax.schedules.Schedule):
    def __init__(self, cycle: int, peak_value: float, min_value: float):
        """Custom cosine cycle schedule."""
        self.cycle = cycle
        self.peak_value = peak_value
        self.min_value = min_value

    def __call__(self, step: int):
        """Compute learning rate at a given step."""
        cycle_pos = step % self.cycle  # Position within the cycle
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * cycle_pos / self.cycle))  # Cosine decay
        return self.min_value + (self.peak_value - self.min_value) * cosine_decay  # Scale


def distribute_dataset(dataset, rank, size):
    """
    Distribute dataset among MPI ranks.

    Args:
    - dataset: List of arrays [pores, conductivities, kappas].
    - rank: Current MPI rank.
    - size: Total number of MPI processes.

    Returns:
    - Local dataset for this rank.
    """
    pores, kappas = dataset

    # Determine chunk size per rank
    n_samples = pores.shape[0]
    chunk_size = n_samples // size

    assert n_samples % size == 0, "Dataset size must be divisible by the number of MPI processes"

    # Compute local data slice
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size

    # Slice the dataset for this rank
    local_pores = pores[start_idx:end_idx]
    local_kappas = kappas[start_idx:end_idx]

    return [local_pores, local_kappas]


def mpi_allreduce_gradients(local_grads, comm):
    # Perform MPI Allreduce to accumulate gradients across all ranks
    return jax.tree_util.tree_map(
        lambda x: comm.allreduce(x, op=MPI.SUM), local_grads
    )


def choose_activation(activation, num_layers):
    activation_functions = []
    if activation == "relu":
        activation_functions = [nnx.relu] * num_layers
    elif activation == "hardtanh":
        activation_functions = [hardtanh] * num_layers
    elif activation == "mixed":
        activation_functions = [nnx.relu] * (num_layers-1) 

        activation_functions.append(hardtanh)

        
    
    return activation_functions


def update_curves(model_name):
    try:
        curves = np.load(f"data/training_results/{model_name}/training_curves.npz", allow_pickle=True)
        n_past_epoch = len(curves['epoch_times'])
    except Exception as e:
        n_past_epoch = 0
    
    return n_past_epoch


def plot_update_learning_curves(exp_name, model_name, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min):
    try:
        curves = np.load(f"experiments/{exp_name}/curves/training_curves_{model_name}.npz", allow_pickle=True)
        
        # Concatenate only new data
        epoch_times_tot = np.concatenate([curves['epoch_times'][:n_past_epoch], epoch_times[:epoch]])
        epoch_losses_tot = np.concatenate([curves['epoch_losses'][:n_past_epoch], epoch_losses[:epoch]])
        valid_losses_tot = np.concatenate([curves['valid_losses'][:n_past_epoch:], valid_losses[:epoch]])
        valid_perc_losses_tot = np.concatenate([curves['valid_perc_losses'][:n_past_epoch:], valid_perc_losses[:epoch]])
        
        # Calculate total epochs
        epoch_tot = len(epoch_losses_tot)
        
        # Plot and save
        #plot_learning_curves(epoch_times_tot, epoch_losses_tot, valid_losses_tot, valid_perc_losses_tot, schedule, model_name, epoch_tot, learn_rate_max, learn_rate_min)
        np.savez(
            f"experiments/{exp_name}/curves/training_curves_{model_name}.npz", 
            epoch_times=epoch_times_tot, 
            epoch_losses=epoch_losses_tot,
            valid_losses=valid_losses_tot, 
            valid_perc_losses=valid_perc_losses_tot,
            allow_pickle=True
        )
    except Exception as e:
        print(f"No training curves file: {e}. Creating new one.")
        #plot_learning_curves(epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, model_name, epoch, learn_rate_max, learn_rate_min)
        np.savez(
            f"experiments/{exp_name}/curves/training_curves_{model_name}.npz", 
            epoch_times=epoch_times, 
            epoch_losses=epoch_losses,
            valid_losses=valid_losses, 
            valid_perc_losses=valid_perc_losses,
            allow_pickle=True
        )


def log_training_progress(model, rank, epoch, n_past_epoch, epochs, avg_loss, avg_val_loss, total_loss_perc, epoch_times):
    if (epoch + 1) % 1 == 0:  # Always true, but keeps the structure flexible
        epoch_str = f"Epoch {epoch + 1 + n_past_epoch}/{epochs + n_past_epoch}, Training Loss: {avg_loss:.2f}, Validation Loss: {avg_val_loss:.2f}, {total_loss_perc:.2f}%, Time: {epoch_times[epoch]:.2f}s"
        
        if not isinstance(model, ensemble) and rank == 0:
            print(epoch_str)
        elif isinstance(model, ensemble):
            if rank == 0:
                print(f"M1: {epoch_str}")
            elif rank == 1:
                print(f"M2: {epoch_str}")
        
        sys.stdout.flush()

def curves_params(exp_name, model_name, model, checkpointer, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min):
    plot_update_learning_curves(exp_name, model_name, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min)
    save_params(exp_name, model_name, model, checkpointer)


