from matinverse import Geometry2D,BoundaryConditions,Fourier
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time

# Define the thermal conductivity map
import numpy as np
import matplotlib.colors as mcolors




def fourier_solver(conductivity):

    # Define geometry
    L = 1
    size = [L, L]
    N = conductivity.shape[1]
    cond = conductivity.reshape((conductivity.shape[0], N**2))

   
    grid = [N, N]

    geo = Geometry2D(grid, size, periodic=[True, True])  
    fourier = Fourier(geo)

    bcs = BoundaryConditions(geo)
    bcs.periodic('y', lambda batch, space, t: 1.0)
    bcs.periodic('x', lambda batch, space, t: 0.0)

    kappa_bulk = jnp.eye(2) 
    # Define kappa as a function
    kappa_map = lambda batch, space, temp, t: kappa_bulk * cond[batch, space]

    output = fourier(kappa_map, bcs, batch_size= cond.shape[0])

    T = output['T']

    T = T.reshape((conductivity.shape))

    # Extract relevant quantities
    kappa_effective = output['kappa_effective']


    return T, kappa_effective


 # Plotting function
def plot_temperature_fourier(Temperatures, base_conductivities, index=0):
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=Temperatures[index].min(), vmax=Temperatures[index].max())

    # Mask pores (low conductivity zones)
    threshold = np.min(base_conductivities[index]) + 1e-6
    masked_T = np.ma.masked_where(base_conductivities[index] < threshold, Temperatures[index])

    plt.figure(figsize=(6, 5))
    im = plt.imshow(masked_T, cmap=cmap, norm=norm, interpolation="nearest")
    plt.colorbar(im, label="Temperature")
    plt.contour(
        masked_T,
        levels=np.linspace(Temperatures[index].min(), Temperatures[index].max(), 25),
        colors="white",
        linewidths=0.5,
    )

    plt.title(f"Heatmap of T (Index {index}) with Level Sets (Fourier)")
    plt.xlabel("x direction")
    plt.ylabel("y direction")
    plt.tight_layout()
    plt.show()


    



if __name__ == "__main__":

    #from utilities_lowfid import test_solver
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    #test_solver(fourier_solver, num_obs=100, name_solver='fourier', fd_check=True)

 

    # Load data
    full_data = jnp.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
    pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
    kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32).flatten()

    # Define 100 target kappa values linearly spaced
    kappa_min, kappa_max = jnp.min(kappas), jnp.max(kappas)
    target_kappas = jnp.linspace(kappa_min, kappa_max, num=500)

    def find_unique_closest_indices(kappas, targets):
        selected = set()
        indices = []
        for t in targets:
            idx = int(jnp.argmin(jnp.abs(kappas - t)))
            if idx not in selected:
                selected.add(idx)
                indices.append(idx)
        return jnp.array(indices)

    selected_indices = find_unique_closest_indices(kappas, target_kappas)

    # Get corresponding pores and true kappas
    selected_pores = pores[selected_indices]
    selected_kappas = kappas[selected_indices]

    print(selected_kappas)

    # Reshape for solver
    selected_pores = selected_pores.reshape((selected_pores.shape[0], 5, 5))
    print(selected_pores.shape)

    from base_conductivity_grid_converter import conductivity_original_wrapper

    grids = conductivity_original_wrapper(selected_pores, 100)

    
    # Run solver
    T, kappa_fourier = fourier_solver(grids)

    
    # Example: plot the first design
    print(kappa_fourier)
    kappa_fourier = kappa_fourier.flatten()

    # Compute errors
    abs_error = jnp.abs(kappa_fourier - selected_kappas)
    percent_error = 100 * (kappa_fourier - selected_kappas) / selected_kappas

    # Print summary
    print(f"Mean Absolute Error: {jnp.mean(abs_error):.4f}")
    print(f"Mean Percentage Error: {jnp.mean(percent_error):.2f}%")

    # --- Combined Plot ---
    fig, ax1 = plt.subplots(figsize=(6, 6))

    # Left y-axis: Fractional Error
    color1 = "tab:blue"
    ax1.scatter(selected_kappas, percent_error, marker="o", linestyle="-", edgecolor="k", color=color1, label="Fractional Error (%)")
    ax1.set_xlabel("Kappa BTE", fontsize=16)
    ax1.set_ylabel("Fractional Error (%)", color=color1, fontsize=16)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)

    # Right y-axis: Kappa Fourier
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.scatter(selected_kappas, kappa_fourier, color=color2, edgecolor="k", label="Kappa Fourier")
    ax2.set_ylabel("Kappa Fourier", color=color2, fontsize=16)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=14)

    ymin = float(min(kappa_fourier) * 0.95)   # small margin below min
    ymax = float(max(kappa_fourier) * 1.05)   # small margin above max
    ax2.set_ylim(ymin, ymax)

    # Add x=y line
    min_val = float(min(jnp.min(selected_kappas), jnp.min(kappa_fourier)))
    max_val = float(max(jnp.max(selected_kappas), jnp.max(kappa_fourier)))
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="x = y")

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=14, loc="best")

    plt.title("BTE vs Fourier Conductivity & Fractional Error", fontsize=18)
    plt.tight_layout()
    plt.savefig("figures/paper/kappa_error_combined.png", dpi=300)
    plt.show()

    # Plot 1: Percentage error vs BTE kappas
    plt.figure(figsize=(5, 5))
    plt.scatter(selected_kappas, percent_error, marker='o', linestyle='-', color='black')
    plt.xlabel("Kappa BTE", fontsize=16)
    plt.ylabel("Fractional Error (%)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.tight_layout()
   
    plt.savefig('figures/paper/fract_error_fourier.png')

    # Plot 2: Scatter plot kappa_Fourier vs kappa_BTE
    plt.figure(figsize=(5, 5))
    plt.scatter(selected_kappas, kappa_fourier, color='blue', edgecolor='k')

    # Add x=y reference line
    min_val = min(jnp.min(selected_kappas), jnp.min(kappa_fourier))
    max_val = max(jnp.max(selected_kappas), jnp.max(kappa_fourier))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.xlabel("Kappa BTE", fontsize=16)
    plt.ylabel("Kappa Fourier", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    #plt.legend(fontsize=14)
    
    plt.savefig('figures/paper/kappa_btevsfourier.png')

    # -----------------------------------------------------------
    # Custom geometry visualization with Fourier solver
    # -----------------------------------------------------------
    import numpy as np
    import matplotlib.colors as mcolors

    # Define your own pore geometry (example: same 5x5 as in HF example)
    pores_custom = np.array([1, 1, 0, 1, 1,
        1, 1, 0, 1, 1,
        0, 1, 1, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 1, 1, 1]).reshape((5, 5))

    # Convert to conductivity grid
    from base_conductivity_grid_converter import conductivity_original_wrapper
    grid_custom = conductivity_original_wrapper(pores_custom[None, :, :], 1000)  # add batch dim

    # Run Fourier solver
    T_custom, kappa_custom = fourier_solver(grid_custom)

    print("Custom geometry effective kappa:", kappa_custom)

   

    # Plot the temperature solution for the custom geometry
    plot_temperature_fourier(T_custom, grid_custom, index=0)

