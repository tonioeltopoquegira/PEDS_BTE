import os
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import optax
from flax import nnx
import pandas as pd
import seaborn as sns

from mpi4py import MPI

cmap = plt.cm.viridis

# Define local data loader
def data_loader(*arrays, batch_size):
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(array[batch_indices] for array in arrays)

def print_generated(model, conductivities, conductivity_res, epoch, model_name, kappa_predicted, kappa_target):
    # Ensure the arrays are NumPy arrays and detach from the computation graph
    conductivities_numpy = [np.asarray(jax.lax.stop_gradient(c)) for c in conductivities[:3]]
    conductivity_res_numpy = [np.asarray(jax.lax.stop_gradient(r)) for r in conductivity_res[:3]]
    kappa_predicted_n = [np.asarray(jax.lax.stop_gradient(r)) for r in kappa_predicted[:3]]

    if model.learn_residual:

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

    else:
        # Create the figure and axes for 3x3 subplots
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [1, 1]})

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
            im2 = axes[i, 1].imshow(conductivity_res_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, 1].set_title('Generated / Final Conductivity')

            # Add text annotations for predicted vs. target values
            fig.text(
                0.90, 0.80 - i * 0.20,  # Adjust position based on row
                f"Predicted: {kappa_predicted_n[i]:.2f}\nTarget: {kappa_target[i]:.2f}",
                fontsize=10, color='black', ha='left', va='center'
            )

            # Create a single colorbar for all subplots (linked to final conductivity)
        fig.colorbar(im2, ax=axes[:3, :], orientation='vertical', label='Conductivity')

    
    # Save the figure
    plt.savefig(f"figures/models/{model_name}/training_evolution/conductivities_epoch_{epoch}.png")
    plt.close()


def clip_gradients(grads, clip_value=1.0):
    return jax.tree_util.tree_map(lambda g: jnp.clip(g, -clip_value, clip_value), grads)


def plot_learning_curves(epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, model_name, epoch, learn_rate_max, learn_rate_min):
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
    plt.savefig(f"figures/models/{model_name}/learning_curve_{schedule}.png")
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
    plt.savefig(f"figures/models/{model_name}/learning_curve_{schedule}_time.png")
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
    os.makedirs(f"data/training_results/{model_name}", exist_ok=True)
    os.makedirs(f"figures/models/{model_name}", exist_ok=True)
    os.makedirs(f"figures/models/{model_name}/training_evolution", exist_ok=True)
    os.makedirs(f"figures/models/{model_name}/final_validation", exist_ok=True)

def choose_activation(activation):

    if activation == "relu":
        return nnx.relu
    if activation == "hardtanh":
        return hardtanh
    
def final_validation(model, model_name, dataset):
    pores, cond, kappa = dataset
    pores = pores.reshape((pores.shape[0], 25))
    if "PEDS" in model_name:
        kappa_pred, _ = model(pores)
    
    else:
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
    output_dir = f"data/training_results/{model_name}/"
    
    # Save the dataset to a CSV file
    output_path = os.path.join(output_dir, "error_results.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")

    print("Error Validation results")

    results = pd.read_csv(f"data/training_results/{model_name}/error_results.csv")

    # Plot KDE for 'kappa_true' and 'kappa_pred'
    plt.figure(figsize=(10, 6))
    sns.kdeplot(results['kappa_true'], label='Kappa True', color='blue', fill=True, alpha=0.4)
    sns.kdeplot(results['kappa_pred'], label='Kappa Pred', color='orange', fill=True, alpha=0.4)

    # Customize the plot
    plt.title(f'Estimated Probability Distribution of Kappa {model_name}', fontsize=16)
    plt.xlabel('Kappa Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"figures/models/{model_name}/final_validation/prob_density_kappa.png")
    plt.close()

    # Compute overestimation and underestimation
    results['error_type'] = results['kappa_pred'] > results['kappa_true']  # True if overestimated

    # Scatter plot of error type
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=results,
        x='kappa_true',
        y=abs(results['kappa_pred'] - results['kappa_true'])/abs(results['kappa_true']),
        hue='error_type',
        palette={True: 'red', False: 'green'},
        alpha=0.6
    )

    # Customize the plot
    plt.title(f'Error Distribution {model_name} (Overestimatio (red) vs Underestimation (green))', fontsize=16)
    plt.xlabel('Kappa True', fontsize=14)
    plt.ylabel('Absolute Error Magnitude', fontsize=14)
    plt.legend(title='Error Type', fontsize=12)
    plt.grid(alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"figures/models/{model_name}/final_validation/error_distribution.png")
    plt.close()

    # Create bins based on 0.1 percentiles of 'kappa_true'
    results['percentile_bin'] = pd.qcut(results['kappa_true'], q=10, precision=3)

    # Compute the average error for each bin
    error_stats = results.groupby('percentile_bin', observed=False)['error'].mean().reset_index()

    # Plot the average error for each percentile bin
    plt.figure(figsize=(12, 6))
    sns.barplot(data=error_stats, x='percentile_bin', y='error', palette='Blues', hue='percentile_bin', dodge=False)


    # Customize the plot
    plt.title(f'Average Error {model_name}', fontsize=16)
    plt.xlabel('Kappa True Percentile Bin', fontsize=14)
    plt.ylabel('Average Error', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(alpha=0.3)

    plt.legend([], [], frameon=False)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"figures/models/{model_name}/final_validation/binned_error_distribution.png")
    plt.close()


def update_curves(model_name):
    try:
        curves = np.load(f"data/training_results/{model_name}/training_curves.npz", allow_pickle=True)
        n_past_epoch = len(curves['epoch_times'])
    except Exception as e:
        n_past_epoch = 0
    
    return n_past_epoch

def plot_update_learning_curves(model_name, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min):
    try:
        curves = np.load(f"data/training_results/{model_name}/training_curves.npz", allow_pickle=True)
        
        # Concatenate only new data
        epoch_times_tot = np.concatenate([curves['epoch_times'][:n_past_epoch], epoch_times[:epoch]])
        epoch_losses_tot = np.concatenate([curves['epoch_losses'][:n_past_epoch], epoch_losses[:epoch]])
        valid_losses_tot = np.concatenate([curves['valid_losses'][:n_past_epoch:], valid_losses[:epoch]])
        valid_perc_losses_tot = np.concatenate([curves['valid_perc_losses'][:n_past_epoch:], valid_perc_losses[:epoch]])
        
        # Calculate total epochs
        epoch_tot = len(epoch_losses_tot)
        
        # Plot and save
        plot_learning_curves(epoch_times_tot, epoch_losses_tot, valid_losses_tot, valid_perc_losses_tot, schedule, model_name, epoch_tot, learn_rate_max, learn_rate_min)
        np.savez(
            f"data/training_results/{model_name}/training_curves.npz", 
            epoch_times=epoch_times_tot, 
            epoch_losses=epoch_losses_tot,
            valid_losses=valid_losses_tot, 
            valid_perc_losses=valid_perc_losses_tot,
            allow_pickle=True
        )
    except Exception as e:
        print(f"No training curves file: {e}. Creating new one.")
        plot_learning_curves(epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, model_name, epoch, learn_rate_max, learn_rate_min)
        np.savez(
            f"data/training_results/{model_name}/training_curves.npz", 
            epoch_times=epoch_times, 
            epoch_losses=epoch_losses,
            valid_losses=valid_losses, 
            valid_perc_losses=valid_perc_losses,
            allow_pickle=True
        )

