from matinverse import Geometry2D,BoundaryConditions,Fourier
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time

# Define the thermal conductivity map



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


    



if __name__ == "__main__":

    from utilities_lowfid import test_solver
    test_solver(fourier_solver, num_obs=100, name_solver='fourier', fd_check=True)

    """N = 100
    conductivity = 160.0 * jnp.ones((2, N, N))
   
    T, kappa = fourier_solver(conductivity)
    T = T[0, :]
    # Plot the heatmap
    plt.figure(figsize=(6, 5))  # Set figure size
    plt.imshow(T, cmap='viridis', origin='lower', extent=[0, N, 0, N])
    plt.colorbar(label="Temperature")  # Add color bar
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Temperature Distribution")
    plt.show()



    

    # Print results
    print("Computed kappa_effective:\n", kappa)"""






