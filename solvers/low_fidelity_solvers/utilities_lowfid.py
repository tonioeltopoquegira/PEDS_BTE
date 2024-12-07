import jax
import jax.random as random
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import jax.lax as lax


# Utilities of the low-fidelity solvers

#Comments ToDO:


def plot_temperature(base_conductivities, Temperatures):

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
    plt.show()



import time

def test_solver(solver, num_obs, jax_inputs=False):

    if jax_inputs:
        full_data = jnp.load("data/highfidelity/high_fidelity_10012_100steps.npz", allow_pickle=True)

        pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
        kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
        base_conductivities = jnp.asarray(full_data['conductivity'], dtype=jnp.float32)

        """
        pores0 = jnp.zeros((1,5,5))
        kappas0 = jnp.array([150.0])
        base0 = jnp.ones((100,100))*150.0

        # Append the 0 observation as first of the dataset
        pores = jnp.vstack([pores0, pores])  # Add pores0 at the beginning
        kappas = jnp.hstack([kappas0, kappas])  # Add kappas0 at the beginning
        base_conductivities = jnp.vstack([base0[None, :, :], base_conductivities])  # Add base0 as the first 2D array

        # Perform forward pass and check for 3
        Temperatures = solver(base_conductivities[:3])
        plot_temperature(base_conductivities, Temperatures)

        """

        # Perform forward and measure speed for num_obs observations
        t = time.time()
        temperatures = solver(base_conductivities[:num_obs])
        print(f"Final time: {time.time()-t} after computing {num_obs} observations on step size 100")


    else:
        
        full_data = np.load("data/highfidelity/high_fidelity_10012_100steps.npz", allow_pickle=True)

        pores = np.asarray(full_data['pores'], dtype=jnp.float32)
        kappas = np.asarray(full_data['kappas'], dtype=jnp.float32)
        base_conductivities = np.asarray(full_data['conductivity'], dtype=jnp.float32)

        pores0 = np.zeros((1,5,5))
        kappas0 = 150.0
        base0 = np.ones((1, 100,100))*150.0

        # Append the 0 observation as first of the dataset
        pores = np.vstack([pores0, pores])  # Add pores0 at the beginning
        kappas = np.hstack([kappas0, kappas])  # Add kappas0 at the beginning
        base_conductivities = np.vstack([base0, base_conductivities])  # Add base0 as the first 2D array

        # Perform forward pass and check for 3
        Temperatures, L = solver(base_conductivities[:3]/150.0)
        #plot_temperature(base_conductivities, Temperatures)

        # Perform forward and measure speed for num_obs observations
        t = time.time()
        #temperatures = solver(base_conductivities[:num_obs])
        print(f"Final time: {time.time()-t} after computing {num_obs} observations on step size 100")

        print("Check Gradients")
        base0 = np.ones((4,100,100))
        base0[:, 1,1] = 0.0
        def dummy_loss(base0):
            x = solver(base0)
            return jnp.sum(x ** 2)


        value, grads = jax.value_and_grad(dummy_loss)(base0)

        print(grads)

