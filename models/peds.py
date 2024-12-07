import jax.numpy as jnp
from flax import nnx
import jax
from jax import lax
import time
import matplotlib.pyplot as plt
from jax import profiler


# SBUROOOOOOOOOOOOOOOOO


# Import components
from models.mlp import mlp
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid

# An easy PEDS Wrapper
class PEDS(nnx.Module):

    def __init__(self, rngs:nnx.Rngs):

        super().__init__()

        self.step_size = 1
        self.generator = mlp(input_size= 25, hidden_sizes=[5], step_size=self.step_size, rngs=rngs)
        self.lowfidsolver = lowfid(iterations=5000)


        # Base Grid generation
        self.N = int(100 / self.step_size)
        self.size_square = int(10 * 1 / self.step_size)
        self.half_size_square = self.size_square // 2
        self.subgrid = jnp.ones((self.size_square, self.size_square)) * 1e-9
        self.indices = jnp.stack(jnp.meshgrid(jnp.arange(5), jnp.arange(5)), axis=-1).reshape(-1, 2)

    def __call__(self, x):

        """
        Takes geometry parametrization and outputs thermal conductivity 
        """
        t_gen = time.time()
        conductivity_res = nnx.vmap(self.generator)(x)
        print("Generation time:", time.time()-t_gen)
        base_t = time.time()
        
        #conductivity_base = base_conductivity_grid(x, self.step_size)
        
        conductivity_base = self.optimized_conductivity_grid(x)
        
        print("Base grid:", time.time()-base_t)
        solv_t = time.time()
        kappa = self.lowfidsolver(conductivity_res+conductivity_base) 
        print("Solver", time.time()- solv_t)

        return kappa
    


    def optimized_conductivity_grid(self, pores):
    
        pores = pores.reshape(pores.shape[0], 5, 5)
        batch_size = pores.shape[0]
        conductivity = jnp.ones((batch_size, self.N, self.N)) * 150.0
        
        start_time = time.time()
        for idx in self.indices:
            x_idx, y_idx = idx

            # Compute the start and end positions for the slice
            start_x = self.half_size_square + x_idx * self.size_square * 2
            start_y = self.half_size_square + y_idx * self.size_square * 2

            # Identify which batches to update
            mask = pores[:, x_idx, y_idx]  # Shape: (batch_size,)
            
            # Generate a full grid of subgrid positions for all batches
            update = mask[:, None, None] * self.subgrid[None, :, :]  # Shape: (batch_size, size_square, size_square)

            # Vectorized update of the conductivity grid
            conductivity = conductivity.at[
                :, start_x : start_x + self.size_square, start_y : start_y + self.size_square
            ].set(jnp.where(mask[:, None, None], update, conductivity[:, start_x : start_x + self.size_square, start_y : start_y + self.size_square]))
        print("For Loop generation time:", time.time() - start_time)
        
        return conductivity

    #nnx.jit(fun=optimized_conductivity_grid, static_argnums=(0, 2))

def base_conductivity_grid(pores, step_size):
    #pores = pores.reshape((5,5))
    N = int(100 / step_size)
    size_square = int(10 * 1 / step_size)
    half_size_square = size_square // 2
    # Initialize the conductivity grid
    conductivity = jnp.ones((N, N)) * 150.0

    # Define an update function for scanning
    def update_grid(cond, idx):
        num_x, num_y = idx

        # Compute the start and end positions for the slice
        start_x = half_size_square + num_x * size_square * 2
        start_y = half_size_square + num_y * size_square * 2

        # Small conductivity value grid to insert
        subgrid = jnp.zeros((size_square, size_square)) + 1e-9

        # Use a mask to apply the subgrid update conditionally
        cond = lax.cond(
            pores[num_x, num_y],  # Condition to check for pore
            lambda c: lax.dynamic_update_slice(c, subgrid, (start_x, start_y)),
            lambda c: c,
            cond
        )
        return cond, None

    # Generate all possible indices and scan over them
    indices = jnp.stack(jnp.meshgrid(jnp.arange(5), jnp.arange(5)), axis=-1).reshape(-1, 2)
    conductivity, _ = lax.scan(update_grid, conductivity, indices)

    return conductivity

jax.jit(fun=base_conductivity_grid, static_argnums=1)
nnx.vmap(f=base_conductivity_grid, in_axes=(0,None))





def plot_three_heatmaps(conductivity_res, conductivity_base):
    """
    Plots heat maps for the base conductivity, generated conductivity, and their sum.

    Arguments:
    - conductivity_res: Generated conductivity grid, shape [N, N].
    - conductivity_base: Base conductivity grid, shape [N, N].
    """
    # Compute the sum
    conductivity_sum = conductivity_res + conductivity_base

    # Plot the three heat maps side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Base conductivity
    im1 = axes[0].imshow(conductivity_base, cmap='viridis', aspect='auto')
    axes[0].set_title('Base Conductivity')
    fig.colorbar(im1, ax=axes[0])

    # Generated conductivity
    im2 = axes[1].imshow(conductivity_res, cmap='viridis', aspect='auto')
    axes[1].set_title('Generated Conductivity')
    fig.colorbar(im2, ax=axes[1])

    # Sum of conductivities
    im3 = axes[2].imshow(conductivity_sum, cmap='viridis', aspect='auto')
    axes[2].set_title('Sum of Conductivities')
    fig.colorbar(im3, ax=axes[2])

    # Add labels
    for ax in axes:
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

    plt.tight_layout()
    plt.show()



