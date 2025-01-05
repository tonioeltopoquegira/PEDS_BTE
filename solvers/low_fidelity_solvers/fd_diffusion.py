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
    h = 5.0  
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
    conductivity = conductivity #/ 150.0
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

    return T * 150.0


def fd_fwd(conductivity):
    T = fd_diffusion(conductivity)
    return T, (conductivity, T)


def fd_bwd(res, grads):

    # prova output costante e vedi se funziona come ti aspetti

    K, T = res
    batch_size, N_cells, _ = K.shape
    dl_dT = grads
    L, L_base, C = laplacian(K)
    

    # We have a loss function l(f(T)), temperature T, conductivity K, laplacian L, source term S
    # Forward we have (K, S) -(1)-> L(K)T=S -(2)-> f(T, K) -(3)-> l(f(T))
    # Backward for the step (2) we want dl/dK = dl/df @ df/dK + dl/df @ df/dT @ L^{-1} @ (dS/dK + dL/dK @ T)  (recall--> T = L^{-1}S)
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
    #L_transpose = L.toarray().transpose()
    lambd = spsolve(L.T, rhs.T) 

    # lambda reshape (batch_size, 10000) ---> (batch_size, 100, 100)
    lambd_reshape = lambd.reshape(K.shape)

    # (batch_size, 100, 100) = (batch_size, 100, 100) @ (batch_size, 100, 100)
    #dl_dK = dl_dT @ lambd_reshape
    dl_dK =  dl_dT @ lambd_reshape 

    #return (lambd_reshape,)
    return (dl_dK,)

# careful with the transposes above !!


fd_diffusion.defvjp(fd_fwd, fd_bwd)


if __name__ == "__main__":

    from utilities_lowfid import test_solver

    test_solver(fd_diffusion, 200, "laplacian")

    import numpy as np
    from jax import numpy as jnp, grad, random
    import matplotlib.pyplot as plt

    # Function to compute numerical gradients using finite differences
    def finite_diff_grad(f, x, eps=1e-4):
        grad = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            x_pos = x.copy()
            x_neg = x.copy()
            x_pos[idx] += eps
            x_neg[idx] -= eps
            grad[idx] = (f(x_pos) - f(x_neg)) / (2 * eps)
        return grad

    # Randomized dot product test for adjoint consistency
    def randomized_dot_product_test(jac_fn, adjoint_fn, x, v, u):
        """
        Tests if ⟨v, J u⟩ == ⟨Jᵀ v, u⟩ where J is the Jacobian.
        """
        # Compute forward product ⟨v, J u⟩
        J_u = jac_fn(x) @ u
        forward_dot = np.dot(v.flatten(), J_u.flatten())
        
        # Compute reverse product ⟨Jᵀ v, u⟩
        J_T_v = adjoint_fn(x, v)
        reverse_dot = np.dot(J_T_v.flatten(), u.flatten())
        
        assert np.isclose(forward_dot, reverse_dot, atol=1e-5), \
            f"Adjoint consistency test failed! Forward: {forward_dot}, Reverse: {reverse_dot}"
        print("Randomized dot product test passed!")
    
    # Visualization function
    def visualize_grad_comparison(fd_grad, custom_grad):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        fd_grad = np.reshape(fd_grad, (9,9))
        custom_grad = np.reshape(custom_grad, (9,9))


        # Heatmap for finite difference gradients
        im1 = axes[0].imshow(fd_grad, cmap="viridis", aspect="auto")
        axes[0].set_title("Finite Difference Gradient")
        plt.colorbar(im1, ax=axes[0])

        # Heatmap for custom gradients
        im2 = axes[1].imshow(custom_grad, cmap="viridis", aspect="auto")
        axes[1].set_title("Custom Gradient")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()

    # Main test function
    def test_fd_diffusion(fd_diffusion):
        """
        Tests the correctness of the fd_diffusion adjoint implementation.
        """
        batch_size = 1
        N_cells = 10
        key = random.PRNGKey(42)
        conductivity = random.uniform(key, shape=(batch_size, N_cells - 1, N_cells - 1))

        # Define a simple scalar loss function
        def loss_fn(cond):
            T = fd_diffusion(cond)
            return jnp.sum(T**2)

        # Finite difference gradient check
        print("Running finite difference gradient check...")
        loss_numpy = lambda x: np.array(loss_fn(jnp.array(x)))
        fd_grad = finite_diff_grad(loss_numpy, np.array(conductivity), eps = 0.001)

        # Custom adjoint gradient
        custom_grad_fn = grad(loss_fn)
        custom_grad = np.array(custom_grad_fn(conductivity))
        visualize_grad_comparison(fd_grad, custom_grad)

        # Compare finite difference and custom gradients
        #assert np.allclose(fd_grad, custom_grad, atol=1e-5), \
        #    f"Finite difference test failed!\nFD Grad:\n{fd_grad}\nCustom Grad:\n{custom_grad}"
        #print("Finite difference test passed!")

        # Randomized dot product test
        print("Running randomized dot product test...")
        v = np.random.randn(*custom_grad.shape)  # Random vector v
        u = np.random.randn(*custom_grad.shape)  # Random vector u

        # Jacobian-vector product function (J u)
        jac_fn = lambda x: grad(loss_fn)(x)

        # Adjoint-vector product function (Jᵀ v)
        adjoint_fn = lambda x, v: grad(lambda cond: jnp.sum(v * fd_diffusion(cond)))(x)

        randomized_dot_product_test(jac_fn, adjoint_fn, conductivity, v, u)

    test_fd_diffusion(fd_diffusion)
    print("All tests passed!")




