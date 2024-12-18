import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time


# Utilities of the low-fidelity solvers



#Comments ToDO:


def plot_temperature(base_conductivities, Temperatures, name_solver):

    cmap = plt.cm.viridis
    norm_T = mcolors.Normalize(vmin=Temperatures[0].min(), vmax=Temperatures[0].max())

    threshold = jnp.min(base_conductivities[:3]) +0.01

    masked_T0 = np.ma.masked_where(base_conductivities[0] < threshold, Temperatures[0, :, :])
    masked_T1 = np.ma.masked_where(base_conductivities[1] < threshold, Temperatures[1, :, :])
    masked_T2 = np.ma.masked_where(base_conductivities[2] < threshold, Temperatures[2, :, :])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # First subplot for Temperature
    im1 = axes[0].imshow(masked_T0, cmap=cmap, norm=norm_T, interpolation='nearest')
    axes[0].set_title('Heatmap of T with Level Sets')
    axes[0].set_xlabel('x direction')
    axes[0].set_ylabel('y direction')
    fig.colorbar(im1, ax=axes[0], label='Temperature')

    contour_levels = np.linspace(Temperatures[0].min(), Temperatures[0].max(), 25)
    axes[0].contour(masked_T0, levels=contour_levels, colors='white', linewidths=0.5)

    # Second subplot for Heat Flux along x-direction
    norm_T1 = mcolors.Normalize(vmin=Temperatures[1].min(), vmax=Temperatures[1].max())
    im2 = axes[1].imshow(masked_T1, cmap=cmap, norm=norm_T1, interpolation='nearest')
    axes[1].set_title('Heatmap of T with Level Sets')
    axes[1].set_xlabel('x direction')
    axes[1].set_ylabel('y direction')
    fig.colorbar(im2, ax=axes[1], label='Temperature')

    contour_levels1 = np.linspace(Temperatures[1].min(), Temperatures[1].max(), 25)
    axes[1].contour(masked_T1, levels=contour_levels1, colors='white', linewidths=0.5)

    # Third subplot for Heat Flux along y-direction
    norm_T2 = mcolors.Normalize(vmin=Temperatures[2].min(), vmax=Temperatures[2].max())
    im3 = axes[2].imshow(masked_T2, cmap=cmap, norm=norm_T2, interpolation='nearest')
    axes[2].set_title('Heatmap of T with Level Sets')
    axes[2].set_xlabel('x direction')
    axes[2].set_ylabel('y direction')
    fig.colorbar(im3, ax=axes[2], label='Temperature')

    contour_levels2 = np.linspace(Temperatures[2].min(), Temperatures[2].max(), 25)
    axes[2].contour(masked_T2, levels=contour_levels2, colors='white', linewidths=0.5)

    plt.tight_layout()
    temp_save_path = f"figures/test_solvers/{name_solver}_temperatures_and_flux.png"
    plt.savefig(temp_save_path)
    print(f"Temperature examples saved to {temp_save_path}")

def plot_gradients(base_conductivities, gradients, name_solver):
    print("Inside!!!")
    cmap = plt.cm.RdYlGn  # Red-Green colormap
    gradients = check_gradients(gradients)
    #norm_g = mcolors.Normalize(vmin=gradients[1].min(), vmax=gradients[1].max())

    threshold = jnp.min(base_conductivities[:3]) + 0.01

    masked_g0 = np.ma.masked_where(base_conductivities[0] < threshold, gradients[0, :, :])
    masked_g1 = np.ma.masked_where(base_conductivities[1] < threshold, gradients[1, :, :])
    masked_g2 = np.ma.masked_where(base_conductivities[2] < threshold, gradients[2, :, :])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    print("Before building")
    # First subplot for Gradient 1
    im1 = axes[0].imshow(masked_g0, cmap=cmap, interpolation='nearest')
    axes[0].set_title('Gradient Example 1')
    axes[0].set_ylabel('y direction')
    fig.colorbar(im1, ax=axes[0], label='Gradient')

    # Second subplot for Gradient 2
    im2 = axes[1].imshow(masked_g1, cmap=cmap, interpolation='nearest')
    axes[1].set_title('Gradient Example 2')
    axes[1].set_xlabel('x direction')
    fig.colorbar(im2, ax=axes[1], label='Gradient')

    # Third subplot for Gradient 3
    im3 = axes[2].imshow(masked_g2, cmap=cmap, interpolation='nearest')
    axes[2].set_title('Gradient Example 3')
    fig.colorbar(im3, ax=axes[2], label='Gradient')

    plt.tight_layout()
    print("Saving")
    grad_save_path = f"figures/test_solvers/{name_solver}_gradients_example.png"
    plt.savefig(grad_save_path)
    print(f"Gradient examples saved to {grad_save_path}")
    


def test_solver(solver, num_obs, name_solver):

    full_data = np.load("data/highfidelity/high_fidelity_10012_100steps.npz", allow_pickle=True)

    pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
    kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
    base_conductivities = np.asarray(full_data['conductivity'], dtype=jnp.float32)

    pores0 = jnp.zeros((1,5,5))
    kappas0 = 150.0
    base0 = jnp.ones((1, 100,100))*150.0

    # Append the 0 observation as first of the dataset
    pores = jnp.vstack([pores0, pores])  # Add pores0 at the beginning
    kappas = jnp.hstack([kappas0, kappas])  # Add kappas0 at the beginning
    base_conductivities = jnp.vstack([base0, base_conductivities])  # Add base0 as the first 2D array

    # Perform forward pass and check for 3
    Temperatures = solver(base_conductivities[:3]/150.0)
    plot_temperature(base_conductivities, Temperatures, name_solver)

    # Perform forward and measure speed for num_obs observations
    t = time.time()
    temperatures = solver(base_conductivities[:num_obs])
    print(f"Final time: {time.time()-t} after computing {num_obs} observations on step size 100")

    print("Check Gradients")
    # Measure time for backward computation with all examples
    @jax.jit
    def loss(base):
        
        Ts = solver(base)
        Jy = jnp.zeros_like(Ts)
        Jy = -base[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / 1.0
        #Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

        kappa_pred = jnp.sum(Jy[:, base.shape[1] // 2, :], axis=-1)
        return jnp.sum((kappa_pred - kappas[:3]) ** 2)
        #return jnp.sum(Ts**2) or jnp.sum(Ts) # Interesting for gradient evaluation

    t_backward = time.time()
    value, grads = jax.value_and_grad(loss)(base_conductivities[:3])
    t_backward = time.time() - t_backward
    print(f"Backward computation time: {t_backward} seconds.")

    # Plot and save the first three gradients
    plot_gradients(base_conductivities, grads[:3].squeeze(), name_solver)
    

def check_gradients(gradients):
    """Check gradients for bad values (NaNs or infinities)."""
    if jnp.any(jnp.isnan(gradients)):
        print("Warning: Gradients contain NaN values.")
    if jnp.any(jnp.isinf(gradients)):
        print("Warning: Gradients contain Inf values.")
    return gradients

