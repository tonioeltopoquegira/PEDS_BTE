import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import jax
import matplotlib.pyplot as plt

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

    L = -(L_batch.T @ C_batch @ L_batch)

    """L_base_coo = L_batch.tocoo()
    plt.scatter(L_base_coo.col, L_base_coo.row, s=1, c='black', marker='.')
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.title("Inside Forward L_batch")
    plt.gca().invert_yaxis()  # Invert y-axis to match matrix representation
    plt.show() """

    return L, Kx, Ky

@jax.custom_vjp
def fd_diffusion(conductivity):

    batch_size, N_cells, _ = conductivity.shape 

    L, _, _ = laplacian(conductivity)

    #assert np.allclose(L[0].toarray(), L[0].toarray().transpose()), "L is not symmetric!"
    #print("Assertion passed! L is Self-Adjoint!!")
    
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

    batch_size, n, _ = K.shape

    L, Kx, Ky = laplacian(K)

    lambd = spsolve(L.transpose(), dl_dT.flatten())

    lambd = np.reshape(lambd, K.shape)

    dL_dKy = -(Ky.T @ Ky)
    dL_dKx = (Kx.T @ Kx)

    dL_dKy = sp.block_diag([dL_dKy] * batch_size, format = 'csr')
    dL_dKx = sp.block_diag([dL_dKx] * batch_size, format = 'csr')


    rhs_1 = dL_dKy @ T.flatten() 

    rhs_0 = dL_dKx @ T.flatten()

    rhs_0 = rhs_0.reshape(K.shape)
    rhs_1 = rhs_1.reshape(K.shape)

    df_dK = - lambd * (rhs_1+rhs_0)

    return ( df_dK,)




fd_diffusion.defvjp(fd_fwd, fd_bwd)


if __name__ == "__main__":

    from utilities_lowfid import test_solver

    test_solver(fd_diffusion, 200, "laplacian", fd_check = True)