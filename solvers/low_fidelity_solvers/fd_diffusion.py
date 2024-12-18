import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import jax
from jax.experimental.sparse import BCSR
from jax.scipy.sparse.linalg import cg


def laplacian(conductivity, h=1.0):
    """
    Computes the discrete Laplacian using the Kronecker product.

    Args:
        conductivity (np.array): (N-1)x(N-1) array with mean conductivity for every cell of the grid.
        h (float): Grid spacing.

    Returns:
        scipy.sparse.csr_matrix: Sparse Laplacian matrix.
    """
    
    #N_cells = conductivity.shape[0]  # Number of cells in the grid
    batch_size, N_cells, _ = conductivity.shape 
    N = N_cells + 1  # Number of points (including boundaries)

    # Define the step sizes (dx1 for forward and dx2 for backward)
    dx1 = np.ones(N - 1)  # Forward step
    dx1[-1] = 0
    dx2 = -np.ones(N-2) # Backward step

    # Create sparse matrix for Laplacian
    D = sp.diags([dx1, dx2], [0, 1], shape=(N - 1, N - 1), format="csr")

    # Scale by h^2 for the Laplacian (if required)
    h = 1.0  
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

    """conductivity[:, -1, :] = 1.0
    conductivity[:, 0,:] = 1.0"""
    conductivity_new = conductivity.at[:, -1, :].set(1.0)
    conductivity_new = conductivity_new.at[:,0,:].set(1.0)
    # Flatten
    conductivity_flat = conductivity_new.reshape(batch_size, N_cells ** 2)
    
    # tile to make it double 
    conductivity_flat = np.concatenate([conductivity_flat, conductivity_flat], axis=1).flatten()
    
    C_batch = sp.diags(conductivity_flat, 0, format="csr")

    # Compute and return the final result: -L.T @ C @ L
    return -(L_batch.T@ C_batch @ L_batch)

@jax.custom_vjp
def fd_diffusion(conductivity):

    batch_size, N_cells, _ = conductivity.shape 
    N = N_cells + 1 # Number of points in the square grid

    x, y = np.linspace(-50, 50, N), np.linspace(-50, 50, N) # in nanometers
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    L = laplacian(conductivity)
    
    # Source term calculation
    S = np.zeros((batch_size, N - 1, N - 1))
    S[:, 0, :] = 0.5 
    S[:, -1, :] = -0.5

    # Print Source term
    S_flat = S.flatten()
   

    T_flat = spsolve(L, S_flat)
    
    # Reshape the solution to 2D
    T = T_flat.reshape((batch_size, N - 1, N - 1))

    return T


def fd_fwd(conductivity):
    T = fd_diffusion(conductivity)
    return T, conductivity

def fd_bwd(res, grads):

    conductivity = res
    dLoss_dT = grads

    L = laplacian(conductivity)

    dLossdT_flatten = dLoss_dT.flatten()

    lambd = spsolve(L.T, dLossdT_flatten.T)


    lambd = lambd.reshape(conductivity.shape)

    #dL_dK = (lambd.T @ dL_dK) @ T  
    
    return (lambd,) 

fd_diffusion.defvjp(fd_fwd, fd_bwd)


if __name__ == "__main__":

    from utilities_lowfid import test_solver

    test_solver(fd_diffusion, 200)




