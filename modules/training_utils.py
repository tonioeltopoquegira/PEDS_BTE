import os
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import optax
from flax import nnx
import pandas as pd

from mpi4py import MPI

cmap = plt.cm.viridis

# Define local data loader
def data_loader(*arrays, batch_size):
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(array[batch_indices] for array in arrays)

def print_generated(conductivities, conductivity_res, epoch, ckpt_dir, kappa_predicted, kappa_target):
    # Ensure the arrays are NumPy arrays and detach from the computation graph
    conductivities_numpy = [np.asarray(jax.lax.stop_gradient(c)) for c in conductivities[:3]]
    conductivity_res_numpy = [np.asarray(jax.lax.stop_gradient(r)) for r in conductivity_res[:3]]
    kappa_predicted_n = [np.asarray(jax.lax.stop_gradient(r)) for r in kappa_predicted[:3]]

    # Create the figure and axes for 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), gridspec_kw={"width_ratios": [1, 1, 1]})

    # Define the range for the color map to ensure all plots use the same scale
    vmin = min(np.min(c) for c in conductivities_numpy + conductivity_res_numpy)
    vmax = max(np.max(c) for c in conductivities_numpy + conductivity_res_numpy)

    # Plot the conductivities and residuals in the subplots
    for i in range(3):
        # Base conductivity
        axes[i, 0].imshow(conductivities_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 0].set_title('Base Conductivity')
        if i == 1:
            axes[i, 0].set_ylabel('y direction')

        # Generated conductivity
        axes[i, 1].imshow(conductivity_res_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 1].set_title('Generated Conductivity')

        # Final conductivity (sum of base and residual)
        im3 = axes[i, 2].imshow(conductivity_res_numpy[i] + conductivities_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 2].set_title('Final Conductivity')

        # Add text annotations for predicted vs. target values
        fig.text(
            0.90, 0.80 - i * 0.20,  # Adjust position based on row
            f"Predicted: {kappa_predicted_n[i]:.2f}\nTarget: {kappa_target[i]:.2f}",
            fontsize=10, color='black', ha='left', va='center'
        )

    # Create a single colorbar for all subplots (linked to final conductivity)
    fig.colorbar(im3, ax=axes[:, :3], orientation='horizontal', label='Conductivity')


    # Save the figure
    plt.savefig(f"figures/{ckpt_dir}/training_evolution/conductivities_epoch{epoch}.png")
    plt.close()


def clip_gradients(grads, clip_value=1.0):
    return jax.tree_util.tree_map(lambda g: jnp.clip(g, -clip_value, clip_value), grads)


def plot_learning_curves(epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, ckpt_dir, epoch, learn_rate_max, learn_rate_min):
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

    # Highlight the epoch where valid_perc_loss hits 5%
    for i, perc_loss in enumerate(valid_perc_losses[:epoch]):
        if perc_loss <= 5.0:
            axs[1].scatter(i, perc_loss, color='red', label=f'Hit 5% at Epoch {i}' if 'Hit' not in axs[1].get_legend_handles_labels()[1] else None)
            break  # Stop after marking the first hit

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Validation % Loss')
    axs[1].set_title('Validation % Loss Evolution')
    axs[1].legend()
    axs[1].grid()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"figures/{ckpt_dir}/learning_curve_{schedule}.png")
    plt.close()

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot: Training and validation losses
    axs[0].plot(epochs, epoch_times[:epoch], 'bo-', label='Epoch Speed')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Time (s)')
    axs[0].set_title(f'Time to complete each epoch')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(epoch_times[:epoch], valid_perc_losses[:epoch], 'ro-', label='Mean Percentual error')
    # Highlight the epoch where valid_perc_loss hits 5%
    for i, perc_loss in enumerate(valid_perc_losses[:epoch]):
        if perc_loss <= 5.0:
            axs[1].scatter(i, perc_loss, color='red', label=f'Hit 5% at Epoch {i}' if 'Hit' not in axs[1].get_legend_handles_labels()[1] else None)
            break  # Stop after marking the first hit

    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Mean Percentual Error')
    axs[0].set_title(f'Mean Percentual Error over Time')
    axs[0].legend()
    axs[0].grid()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"figures/{ckpt_dir}/learning_curve_{schedule}_time.png")
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

@nnx.jit
def accumulate_gradients(total_grads, new_grads):
        if total_grads is None:
            return new_grads
        return jax.tree_util.tree_map(lambda x, y: x + y, total_grads, new_grads)

@nnx.jit
def hardtanh(x):
    """Hard tanh activation: max(-1, min(1, x))."""
    return jnp.clip(x, -1, 1)

def choose_schedule(schedule, learn_rate_min, learn_rate_max, epochs):
    
    if schedule == "cosine-decay":
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learn_rate_max,  # Maximum learning rate
            decay_steps=epochs,   # Number of epochs to decay over
            alpha=learn_rate_min / learn_rate_max  # Minimum learning rate as a fraction of max
        )
    if schedule == "cosine-cycles":
        lr_schedule_onecycle = optax.cosine_decay_schedule(
            init_value=learn_rate_max,  # Maximum learning rate
            decay_steps=epochs // 25,   # Number of epochs to decay over
            alpha=learn_rate_min / learn_rate_max  # Minimum learning rate as a fraction of max
        )

        lr_schedules = [lr_schedule_onecycle] * (epochs // 50)
        boundaries = [i * (epochs // 50) for i in range(1, 50)]
        lr_schedule = optax.join_schedules(lr_schedules, boundaries)

    if schedule == "exponential":
        lr_schedule = optax.exponential_decay(
            init_value=learn_rate_max,
            transition_steps=500,
            decay_rate=learn_rate_min / learn_rate_max
        )

    if schedule == "constant":
        lr_schedule = learn_rate_min

    
    return lr_schedule


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
    pores, conductivities, kappas = dataset

    # Determine chunk size per rank
    n_samples = pores.shape[0]
    chunk_size = n_samples // size

    assert n_samples % size == 0, "Dataset size must be divisible by the number of MPI processes"

    # Compute local data slice
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size

    # Slice the dataset for this rank
    local_pores = pores[start_idx:end_idx]
    local_conductivities = conductivities[start_idx:end_idx]
    local_kappas = kappas[start_idx:end_idx]

    return [local_pores, local_conductivities, local_kappas]

def mpi_allreduce_gradients(local_grads, comm):
    # Perform MPI Allreduce to accumulate gradients across all ranks
    return jax.tree_util.tree_map(
        lambda x: comm.allreduce(x, op=MPI.SUM), local_grads
    )

def create_folders(model_name):
    os.makedirs(f"data/results/{model_name}", exist_ok=True)
    os.makedirs(f"figures/{model_name}", exist_ok=True)
    os.makedirs(f"figures/{model_name}/training_evolution", exist_ok=True)
    os.makedirs(f"figures/{model_name}/final_validation", exist_ok=True)

def choose_activation(activation):

    if activation == "relu":
        return nnx.relu
    if activation == "hardtanh":
        return hardtanh
    
def final_validation(model, model_name, dataset):
    pores, cond, kappa = dataset
    pores = pores.reshape((pores.shape[0], 25))
    kappa_pred = model(pores)
    kappa_pred = kappa_pred.squeeze(-1)
    error = np.abs(kappa_pred - kappa) / np.abs(kappa_pred)

    # Create a DataFrame with results
    results_df = pd.DataFrame({
        "pores": list(pores),
        "kappa_true": kappa,
        "kappa_pred": kappa_pred,
        "error": error
    })

    # Define the output directory and ensure it exists
    output_dir = f"data/results/{model_name}/"
    
    # Save the dataset to a CSV file
    output_path = os.path.join(output_dir, "error_results.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")