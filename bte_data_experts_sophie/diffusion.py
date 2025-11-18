import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import jax
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
import time
import sparse 

# Solver for periodic boundaries diffusion w/ finite differences and adjoint sensitivity

def laplacian(conductivity, h=1.0):
    """
    Computes the discrete Laplacian using the Kronecker product.

    Args:
        conductivity (np.array): (N-1)x(N-1) array with mean conductivity for every cell of the grid.
        h (float): Grid spacing.

    Returns:
        scipy.sparse.csr_matrix: Sparse Laplacian matrix.
    """
   
    batch_size, N_cells, _ = conductivity.shape 
    N = N_cells + 1  # Number of points (including boundaries)

    # Define the step sizes (dx1 for forward and dx2 for backward)
    dx1 = np.ones(N - 1)  # Forward step
    dx1[-1] = 0
    dx2 = -np.ones(N-2) # Backward step

    # Create sparse matrix for Laplacian
    D = sp.diags([dx1, dx2], [0, 1], shape=(N - 1, N - 1), format="csr")

    # Scale by h^2 for the Laplacian (if required)
    D = D / (h**2) # can we delete it??


    # Create identity matrix for the size of the Laplacian matrix
    I = sp.eye(N - 1, format="csr")

    Kx = sp.kron(D, I, format="csr")
    Ky = sp.kron(I, D, format="csr")

    Ky = Ky.tolil()

    # Set the first and last N_cells rows to 0, except for the diagonals
    Ky[:N_cells, :] = 0
    Ky[-N_cells:, :] = 0

    # Set the diagonal elements to 1 (where necessary)
    Ky[np.arange(N_cells), np.arange(N_cells)] = 1
    Ky[np.arange(N_cells*(N_cells-1), N_cells**2), np.arange(N_cells*(N_cells-1), N_cells**2)] = 1


    # Convert back to CSR format for efficient computation
    Ky = Ky.tocsr()

    # Stack Kx and Ky vertically to form the 2D Laplacian
    L = sp.vstack([Kx, Ky])


    # repeat L batch_size times to have a final sparse array of size (batch_size, L.shape[0], L.shape[1])
    L_batch = sp.block_diag([L] * batch_size, format="csr")

    # Flatten
    conductivity_flat = conductivity.reshape(batch_size, N_cells ** 2)
    
    # tile to make it double 
    conductivity_flat = np.concatenate([conductivity_flat, conductivity_flat], axis=1).flatten()
    
    C_batch = sp.diags(conductivity_flat, 0, format="csr")

    L_final = -(L_batch.T @ C_batch @ L_batch)


    return L_final, L

@jax.custom_vjp
def fd_diffusion(conductivity):

    batch_size, N_cells, _ = conductivity.shape 

    L, _ = laplacian(conductivity)
    
    # Source term calculation
    S = np.zeros((batch_size, N_cells, N_cells))
    S[:, 0, :] = -0.5 
    S[:, -1, :] = +0.5

    # Print Source term
    S_flat = S.flatten()
   
    T_flat = spsolve(L, S_flat)
    
    # Reshape the solution to 2D
    T = T_flat.reshape((batch_size, N_cells, N_cells))

    return 150.0 * T


def fd_fwd(conductivity):
    T = fd_diffusion(conductivity)
    return T, (conductivity, T)


def fd_bwd(res, dl_dT):

    K, T = res

    batch_size, N_cells, _ = T.shape
    
    L, L_small = laplacian(K)

    L_repeated_list = [L_small] * batch_size

    L_batch_sparse= sp.vstack(L_repeated_list)

    L_batch_sparse = sparse.COO(L_batch_sparse)

    N_cells_sq = N_cells**2

    L_batch_sparse = L_batch_sparse.reshape((batch_size, 2 * N_cells_sq, N_cells_sq))

    dL_dk_tiled_sparse = -sparse.einsum('bri,brj->bijr', L_batch_sparse, L_batch_sparse)
   
    dL_dk_sparse = dL_dk_tiled_sparse.reshape(( batch_size, N_cells_sq, N_cells_sq, 2, N_cells, N_cells))

    dL_dk_sparse = dL_dk_sparse.sum(axis=3)

    dL_dk_sparse = dL_dk_sparse.reshape((batch_size, N_cells_sq,N_cells_sq, N_cells, N_cells))

    lambd = spsolve(L, dl_dT.flatten())

    lambd = lambd.reshape((batch_size, N_cells_sq))
    T = T.reshape((batch_size, N_cells_sq))

    dfdK_sparse = sparse.einsum('hi,hijab,hj->hab', lambd, dL_dk_sparse, T)
    dfdK_sparse = dfdK_sparse.reshape(K.shape)

    return (-dfdK_sparse.todense(),)


fd_diffusion.defvjp(fd_fwd, fd_bwd)


# Utilities for conductivity grid conversion (i.e. from 25 parameters to NxN grid)

def optimized_conductivity_grid_jax(pores, N):
    step_size = 100 / N
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



def conductivity_grid_5by5(pores, binary=False):
    if binary:
        
        result = jnp.where(pores, 1e-7, 160) 
    else:
        
        result = 160 * (1 - pores)
        #
        result = jnp.clip(result, 1e-7, 160) 

    return jnp.reshape(result, (pores.shape[0], 5, 5))

   
def conductivity_original_wrapper(pores, N, binary=False):

    if N >= 20:

        return optimized_conductivity_grid_jax(pores, N)
    

    if N == 10:

        pass

    if N ==5:

        return conductivity_grid_5by5(pores, binary=binary)
    

# Compute the effective conductivity kappa from the temperature

def flux_kappa(conductivity, Ts):
    # Initialize arrays for fluxes
    Jy = jnp.zeros_like(Ts)
    
    Jy = -conductivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / 1.0
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    kappas = jnp.sum(Jy[:,   conductivity.shape[1] // 2, :], axis=-1)

    return kappas 


# Final wrapper for diffusion expert

def diffusion_expert(params, N):
    # Convert params to conductivity grid
    conductivity = conductivity_original_wrapper(params, N)

    # Solve diffusion equation
    temperature = fd_diffusion(conductivity)

    # Compute effective conductivity kappa from temperature gradient
    kappa = flux_kappa(conductivity, temperature)

    return kappa
    

if __name__ == "__main__":


    kappa = diffusion_expert(jnp.ones((1000,25)), N=100)

    print("Kappa:", kappa)