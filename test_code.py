import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from jax import random, lax
import time
import jax
import numpy as np

step_size = 1
N = int(100 / step_size)

def optimized_conductivity_grid_jax(pores, resolution):
    step_size = int(100/resolution)
    N = int(100/ step_size)
    size_square = int(10 * 1 / step_size)
    half_size_square = size_square // 2
    subgrid = jnp.ones((size_square, size_square)) * 1e-9
    indices = jnp.stack(jnp.meshgrid(jnp.arange(5), jnp.arange(5)), axis=-1).reshape(-1, 2)

    batch_size = pores.shape[0]
    pores = jnp.reshape(pores, [batch_size, 5, 5])
    
    conductivity = jnp.ones((batch_size, N, N)) * 150.0
    
    for idx in indices:
        x_idx, y_idx = idx

        # Compute the start and end positions for the slice
        start_x = half_size_square + x_idx * size_square * 2
        start_y = half_size_square + y_idx * size_square * 2

        # Identify which batches to update
        mask = pores[:, x_idx, y_idx]  # Shape: (batch_size,)
        
        # Generate a full grid of subgrid positions for all batches
        update = mask[:, None, None] * subgrid[None, :, :]  # Shape: (batch_size, size_square, size_square)

        # Vectorized update of the conductivity grid
        conductivity = conductivity.at[
            :, start_x : start_x + size_square, start_y : start_y + size_square
        ].set(jnp.where(mask[:, None, None], update, conductivity[:, start_x : start_x + size_square, start_y : start_y + size_square]))
    #print("For Loop generation time:", time.time() - start_time)
    
    return conductivity




# Ingest data <- Here we will do active learning
"""full_data = jnp.load("data/highfidelity/high_fidelity_10012_20steps.npz", allow_pickle=True)

print(full_data.keys())

pores_data = full_data['pores'][:300]  # Slice the first 10,000 entries

# Reshape the data to have the shape (N, 5, 5), where N is the number of samples
pores_reshaped = pores_data.reshape(pores_data.shape[0], 5, 5)

# Now convert to a JAX array
pores = jnp.asarray(pores_reshaped, dtype=jnp.float32)

check_speed = time.time()
conductivity_50 = optimized_conductivity_grid_jax(pores, resolution=100)
print(f"Speed: {time.time()-check_speed} for {conductivity_50.shape} size")

"""


# CAREFUL THAT THE ORDER IS SAME FOR generated values and this values... i.e. where is
#  the origin (0,0) of the conductivity with respect to the pores matrix [0,1] where 0 is no pore 
# and 1 is pore. I think it might be correct now... just check if i am using "upper" or "lower" there
# they must coincide given that they give same result

# Plot the results
#fig, axes = plt.subplots(1, 2)

# Plot the pores grid


#print(pores[33])
#im = axes[1].imshow(conductivity_50[33], cmap='hot', origin="upper")
#axes[1].set_title('Conductivity Grid')
#axes[1].set_xticks([])
#axes[1].set_yticks([])

# Add colorbar to conductivity grid
#fig.colorbar(im, ax=axes[0])

#plt.tight_layout()
#plt.show()



