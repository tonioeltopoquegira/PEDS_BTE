import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.scipy.sparse import diags
from jax.scipy.sparse import csr_matrix
from jax import random

# Finite Difference Operator for Dirichlet boundaries
def sdiff1(x):
    N = len(x) - 2
    dx1 = 1 / (x[1:N+1] - x[:N])
    dx2 = -1 / (x[1:N+1] - x[:N])
    diagonals = [dx1, dx2]
    offsets = [0, -1]
    D = diags(diagonals, offsets, shape=(N + 1, N))
    return csr_matrix(D)

# Finite Difference Operator for periodic boundaries
def sdiff1_periodic(x):
    N = len(x) - 2
    dx1 = 1 / (x[1:N+1] - x[:N])
    dx2 = -1 / (x[1:N+1] - x[:N])
    diagonals = [dx1, dx2]
    offsets = [0, -1]
    D = diags(diagonals, offsets, shape=(N + 1, N + 1))
    D[-1, -1] = -D[-1, -2]
    D[0, -1] = -D[0, 0]
    return csr_matrix(D)

# Position grid generator
def get_position(L, resolution):
    nx = int(L * resolution)  # Number of points in x
    δ = 1 / resolution
    return jnp.arange(1, nx + 1) * δ

# Compute ∇⋅c∇ (Laplacian) operator for a function c(x, y)
def Laplacian(x, y, c, periodicy=True):
    Dx = sdiff1(x)
    Nx = Dx.shape[1]
    Dy = sdiff1_periodic(y) if periodicy else sdiff1(y)
    
    # Discrete gradient operator
    G = jnp.kron(Dx.toarray(), jnp.eye(Ny)) + jnp.kron(jnp.eye(Nx), Dy.toarray())
    
    # Grids for derivatives in x and y directions
    xp = 0.5 * (x[:-1] + x[1:])
    yp = 0.5 * (y[:-1] + y[1:])
    
    # Evaluate c(x)
    if periodicy:
        C = diags(c, 0)
    else:
        # Handle Dirichlet boundaries (adjust as necessary)
        C = diags(c, 0)
    
    return -G.T @ C @ G  # ∇⋅c∇

# Target function for the diffusion solver
def targetfunc(x, y, c):

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    A = Laplacian(x, y, c, periodicy=True)
    
    n = len(x)
    c1 = c[:(n - 1) * (n - 1)]
    c1 = c1.reshape((n - 1, n - 1))
    S = jnp.zeros((len(x) - 2, len(y) - 1))
    S = S.at[-1, :].set(-c1[-1, :] / dx**2)
    
    T = jnp.linalg.solve(A.toarray(), S.ravel()).reshape((len(x) - 2, len(y) - 1))
    iline = jnp.sum(x < 0.5)  # Find index for x < 0.5
    integrand = c1[iline, :] * (T[iline + 1, :] - T[iline, :])
    
    return jnp.sum(integrand) / (dx * dy)

