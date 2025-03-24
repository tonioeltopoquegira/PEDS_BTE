import numpy as np
import scipy.linalg
from numpy.polynomial.hermite import hermgauss

# Choose better co-location + batch + covariances???

def pce(K_mean, K_cov, pde_solver, order=3, quadrature_points=5):
    """
    Propagates uncertainty using Polynomial Chaos Expansion (PCE) with collocation points.

    Parameters:
    - K_mean (NxN array): Mean of the input matrix K.
    - K_cov ((N^2)x(N^2) array): Covariance matrix of flattened K.
    - pde_solver (function): Function that takes an NxN matrix K and returns an NxN matrix T.
    - order (int): Order of the PCE expansion.
    - quadrature_points (int): Number of Gauss-Hermite quadrature points.

    Returns:
    - T_mean (NxN array): Mean of the output T.
    - T_var (NxN array): Variance of the output T.
    """

    N = K_mean.shape[0]  
    num_vars = N * N      
    
    # Compute Gauss-Hermite quadrature points and weights
    xi_vals, weights = hermgauss(quadrature_points)  

    weights /= np.sqrt(np.pi) 
    print(xi_vals)
    print(weights)


    # Sample realizations of K from the covariance matrix
    L = scipy.linalg.cholesky(K_cov, lower=True)  # Cholesky decomposition
    K_samples = [K_mean.flatten() + L @ (xi * np.ones(num_vars)) for xi in xi_vals]  # Perturbed K
    
    # Solve the PDE for each collocation point
    T_samples = np.array([pde_solver(K.reshape(1, N, N)).reshape((N,N)) for K in K_samples])  # Solve PDE at collocation points



    # Compute PCE coefficients using weighted projection
    c_k = np.zeros((order + 1, N, N))
    for k in range(order + 1):
        hermite_poly_k = np.polynomial.hermite.Hermite.basis(k)(xi_vals)  # Hermite polynomial values
        for q in range(quadrature_points):
            c_k[k] += weights[q] * T_samples[q] * hermite_poly_k[q]  # Weighted projection

    # Compute mean and variance
    T_mean = c_k[0]  
    T_var = np.sum(c_k[1:]**2, axis=0)  

    return T_mean, T_var


if __name__ == "__main__":

    from solvers.low_fidelity_solvers.fd_diffusion import fd_diffusion

    N = 10

    K_mean = np.full((N, N), 150.0)  

    
    constant_variance = 25.0  # Example variance value
    increasing_variance = np.array([
    constant_variance * (1 + 4 * (1 - abs(i - N//2) / (N//2))) for i in range(N)
    ])

    # Construct diagonal covariance matrices (no correlations)
    K_cov_constant = np.diag(np.full(N * N, constant_variance))  
    K_cov_variable = np.diag(np.tile(increasing_variance, N))  # Increasing variance along vertical axis

    T_mean, T_cov = pce(K_mean, K_cov_variable, fd_diffusion, order=3, quadrature_points=5)

    print(T_mean)
    print(T_cov)
    
