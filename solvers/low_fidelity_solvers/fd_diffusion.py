import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import jax

def laplacian(conductivity, h=1.0):
    """
    Computes the discrete Laplacian using the Kronecker product.

    Args:
        conductivity (np.array): (N-1)x(N-1) array with mean conductivity for every cell of the grid.
        h (float): Grid spacing.

    Returns:
        scipy.sparse.csr_matrix: Sparse Laplacian matrix.
    """
    #conductivity = lax.stop_gradient(conductivity)
    #conductivity = np.array(conductivity)
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

    #conductivity[:, -1, :] = 1.0
    #conductivity[:, 0,:] = 1.0
    conductivity = conductivity / 150.0
    conductivity_new = conductivity.at[:, -1, :].set(1.0)
    conductivity_new = conductivity_new.at[:,0,:].set(1.0)
    # Flatten
    conductivity_flat = conductivity.reshape(batch_size, N_cells ** 2)
    
    # tile to make it double 
    conductivity_flat = np.concatenate([conductivity_flat, conductivity_flat], axis=1).flatten()
    
    C_batch = sp.diags(conductivity_flat, 0, format="csr")

    L = -(L_batch.T @ C_batch @ L_batch)

    return L, L_batch, C_batch

@jax.custom_vjp
def fd_diffusion(conductivity):

    batch_size, N_cells, _ = conductivity.shape 

    L, _, _ = laplacian(conductivity)
    
    # Source term calculation
    S = np.zeros((batch_size, N_cells, N_cells))
    S[:, 0, :] = -0.5 
    S[:, -1, :] = +0.5

    # Print Source term
    S_flat = S.flatten()
   
    T_flat = spsolve(L, S_flat)
    
    # Reshape the solution to 2D
    T = T_flat.reshape((batch_size, N_cells, N_cells))

    return T


def fd_fwd(conductivity):
    T = fd_diffusion(conductivity)
    return T, (conductivity, T)


"""def fd_bwd(res, grads):

    # prova output costante e vedi se funziona come ti aspetti

    K, T = res
    batch_size, N_cells, _ = K.shape
    dl_dT = grads
    L, L_base, C = laplacian(K)
    

    # We have a loss function l(f(T)), temperature T, conductivity K, laplacian L, source term S
    # Forward we have (K, S) -(1)-> L(K)T=S -(2)-> f(T, K) -(3)-> l(f(T))
    # Backward for the step (2) we want dl/dK = dl/df @ df/dK + dl/dT @ L^{-1} @ (dS/dK + dL/dK @ T)  (recall--> T = L^{-1}S)
    # Note that the first term is already taken in account before in the backprop... we just need to work on:
    # dT/dK = L^{-1} @ ( dS/dK + dL/dK @ T) where dS/dK = 0 therefore we have
    # dT/dK = L^{-1} @ dL/dK @ T OR dl/dK = dl/dT @ dT/dK = dl/dT @ L^{-1} @ ( dS/dK + dL/dK @ T)

    # So we first solve the adjoint equation for lambda = dT/dK 
    # L @ lambda = (dL/dK @ T).T (note that L is self-adjoint so L=L.T)
    # and then multiply 
    # dl/dK = dl/dT @ lambda  

    # Diagonal derivative dC/dK (Identity-like behavior)
    conductivity_flat = K.reshape(batch_size, N_cells ** 2)
    conductivity_flat = np.concatenate([conductivity_flat, conductivity_flat], axis = 1)
    dC_dK = sp.diags(np.ones_like(conductivity_flat.flatten()), 0, format="csr")

    dL_dK = -(L_base.T @ dC_dK @ L_base)

    # (batch_size, 100, 100, 100, 100) @ (batch_size, 100, 100) = (batch_size, 100, 100)
    # (batch_size, 10000, 10000) @ (batch_size, 10000) = (batch_size, 10000)
    rhs = dL_dK @ T.flatten()

    # (batch_size, 10000, 10000) @ lambd = (batch_size, 10000)
    lambd = spsolve(L, rhs)

    # lambda reshape (batch_size, 10000) ---> (batch_size, 100, 100)
    lambd_reshape = lambd.reshape(K.shape)

    # (batch_size, 100, 100) = (batch_size, 100, 100) @ (batch_size, 100, 100)
    dl_dK = dl_dT @ lambd_reshape

    return (dl_dK,)"""

"""def fd_bwd(res, grads):

    K, T = res
    batch_size, N_cells, _ = K.shape
    L, L_base, C = laplacian(K)

    lambd = spsolve(L, grads.flatten())

    conductivity_flat = K.reshape(batch_size, N_cells ** 2)
    conductivity_flat = np.concatenate([conductivity_flat, conductivity_flat], axis = 1)
    dC_dK = sp.diags(np.ones_like(conductivity_flat.flatten()), 0, format="csr")
    dL_dK = -(L_base.T @ dC_dK @ L_base)
    lambd = lambd.reshape(K.shape)

    # Compute dL/dK using λ and T
    #dL_dK = - lambd @ T / 150
    #dL_dK = - lambd * T / 150
    #dL_dK = np.einsum('bij,bjk->bik', lambd, T)

    #dL_dK = - lambd @ dL_dK 

    #dL_dK = dL_dK.reshape(K.shape) @ T
    
    return (dL_dK,)"""

def fd_bwd(res, grads):
    
    conductivity, T = res  # Extract forward pass variables
    batch_size, N_cells, _ = conductivity.shape

    # Recompute the Laplacian and related matrices
    L, L_batch, C_batch = laplacian(conductivity)

    # Flatten gradients from T for solving adjoint equation
    grads_flat = grads.flatten()

    # Solve the adjoint equation: L^T λ = ∂L/∂T (grads)
    adjoint_variable = spsolve(L.T, grads_flat)

    # Compute dL/dC (gradient w.r.t. conductivity matrix)
    dL_dC = (L_batch @ adjoint_variable)

    # Reshape dL/dC back to match the conductivity dimensions
    dL_dC = dL_dC.reshape((batch_size, 2 * (N_cells**2)))


    # Sum over the two tiled conductivity contributions
    dL_dC_reshaped = dL_dC[:, :N_cells**2] + dL_dC[:, N_cells**2:]

    # Adjust for conductivity's placement in sparse matrix
    dL_dK = dL_dC_reshaped.reshape((batch_size, N_cells, N_cells))

    return (dL_dK,)


fd_diffusion.defvjp(fd_fwd, fd_bwd)


if __name__ == "__main__":

    from utilities_lowfid import test_solver

    test_solver(fd_diffusion, 200, "laplacian")




