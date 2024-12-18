import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
cmap = plt.cm.viridis

def data_loader(*arrays, batch_size):
    """
    A simple data loader for batching data arrays.

    Args:
        arrays: One or more JAX arrays with the same first dimension.
        batch_size: The number of samples per batch.

    Yields:
        Batches of data arrays.
    """
    # Ensure all arrays have the same number of samples
    n_samples = arrays[0].shape[0]
    for array in arrays:
        assert array.shape[0] == n_samples, "All input arrays must have the same first dimension."
    
    indices = jnp.arange(n_samples)  # Use jnp.arange for JAX arrays
    
    # Split into batches and yield
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(array[batch_indices] for array in arrays)


def print_generated(conductivities, conductivity_res, epoch):
    # Convert conductivities to numpy for plotting
    base_conductivity_numpy = jax.lax.stop_gradient(conductivities[0])  # detaching from graph
    res_conductivity_numpy = jax.lax.stop_gradient(conductivity_res[0])  # detaching from graph

    # Ensure the arrays are NumPy arrays
    base_conductivity_numpy = np.asarray(base_conductivity_numpy)
    res_conductivity_numpy = np.asarray(res_conductivity_numpy)

    # Create the figure and axes for 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Define the range for the color map to ensure all plots use the same scale
    vmin = min(base_conductivity_numpy.min(), res_conductivity_numpy.min())
    vmax = max(base_conductivity_numpy.max(), res_conductivity_numpy.max(), (res_conductivity_numpy + base_conductivity_numpy).max())

    # First subplot for Base Conductivity
    im1 = axes[0].imshow(base_conductivity_numpy, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].set_title('Base Conductivity')
    axes[0].set_ylabel('y direction')

    # Second subplot for Residual Conductivity
    im2 = axes[1].imshow(res_conductivity_numpy, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1].set_title('Generated Conductivity')
    axes[1].set_xlabel('x direction')

    # Third subplot for Final Conductivity (sum of base and residual)
    im3 = axes[2].imshow(res_conductivity_numpy + base_conductivity_numpy, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[2].set_title('Final Conductivity')

    # Create a single colorbar for all subplots
    fig.colorbar(im3, ax=axes, orientation='horizontal', label='Conductivity')

    # Adjust layout for better spacing
    #plt.tight_layout()

    # Save the figure
    plt.savefig(f"figures/training/base_peds/conductivities_epoch{epoch}.png")
    plt.close()  #


def clip_gradients(grads, clip_value=1.0):
    return jax.tree_util.tree_map(lambda g: jnp.clip(g, -clip_value, clip_value), grads)
