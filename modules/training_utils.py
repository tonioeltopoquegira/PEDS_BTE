import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import optax
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


"""def print_generated(conductivities, conductivity_res, epoch, ckpt_dir):
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

    # Save the figure
    plt.savefig(f"figures/{ckpt_dir}/training_evolution/conductivities_epoch{epoch}.png")
    plt.close()  #"""

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


def plot_learning_curves(epoch_losses, valid_losses, schedule, ckpt_dir, epoch):
    """Plot the learning curve."""
    epochs = jnp.arange(epoch)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses[:epoch], 'bo-', label='Training Loss')
    plt.plot(epochs, valid_losses[:epoch], 'ro-', label = 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/{ckpt_dir}/learning_curve_{schedule}.png")


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
            decay_steps=epochs // 50,   # Number of epochs to decay over
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