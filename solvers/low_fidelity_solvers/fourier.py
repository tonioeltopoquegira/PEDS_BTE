from matinverse import Geometry2D,BoundaryConditions,Fourier

import matplotlib.pyplot as plt
from jax import numpy as jnp

import jax

# Define geometry
L = 1
size = [L, L]
N = 20
grid = [N, N]
batch_size = 1000

geo = Geometry2D(grid, size, periodic=[True, True])  
fourier = Fourier(geo)

bcs = BoundaryConditions(geo)
bcs.periodic('x', lambda batch, space, t: 1.0)
bcs.periodic('y', lambda batch, space, t: 0.0)

# Define the thermal conductivity map
kappa_bulk = jnp.eye(2)  # Base conductivity tensor
key = jax.random.PRNGKey(0)  # Create a random key
rho = jax.random.uniform(key, shape=(batch_size, N**2))

#rho = jnp.ones((1, N**2))

# Define kappa as a function
kappa_map = lambda batch, space, temp, t: kappa_bulk * rho[batch, space]

# Call the Fourier solver
import time
t_temp = time.time()
output = fourier(kappa_map, bcs, batch_size=batch_size)
print(f"{time.time()-t_temp}")

T = output['T']

T = T[0, :]

T = T.reshape((N,N))

# Plot the heatmap
plt.figure(figsize=(6, 5))  # Set figure size
plt.imshow(T, cmap='viridis', origin='lower', extent=[0, N, 0, N])
plt.colorbar(label="Temperature")  # Add color bar
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Temperature Distribution")
plt.show()



# Extract relevant quantities
kappa_effective = output['kappa_effective']

# Print results
print("Computed kappa_effective:\n", kappa_effective)



