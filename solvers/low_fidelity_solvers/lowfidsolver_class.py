from flax import nnx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

from solvers.low_fidelity_solvers.gauss_diffusion import gauss_solver
from solvers.low_fidelity_solvers.fd_diffusion import fd_diffusion
from matinverse import Geometry2D,BoundaryConditions,Fourier

class lowfid(nnx.Module):
   
    def __init__(self, solver="gauss", iterations=5000):
        self.iterations = iterations
        self.solver = solver

        if solver == "fourier":
            L = 1
            size = [L, L]
            N = 20
            grid = [N, N]
            # create bcs and geometries
            geo = Geometry2D(grid, size, periodic=[True, True])  
            self.fourier = Fourier(geo)

            self.bcs = BoundaryConditions(geo)
            self.bcs.periodic('x', lambda batch, space, t: 1.0)
            self.bcs.periodic('y', lambda batch, space, t: 0.0)
           

    def __call__(self, conductivity):

        if self.solver == "fourier":
            # here reshape the conductivities and explicit batch_size
            batch_size, N, _ = conductivity.shape
            conductivity = conductivity.reshape((batch_size, N**2))
            cond_map = lambda batch, space, temp, t: conductivity[batch, space]

            return self.fourier(cond_map, self.bcs, batch_size = batch_size)

        if self.solver == "gauss":
            T = gauss_solver(conductivity)
        
        if self.solver == "direct":
            T = fd_diffusion(conductivity)

        return flux_kappa(conductivity, T)

@nnx.jit
def flux_kappa(conductivity, Ts):
    # Initialize arrays for fluxes
    Jy = jnp.zeros_like(Ts)
    
    Jy = -conductivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / 1.0
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    kappas = jnp.sum(Jy[:,   conductivity.shape[1] // 2, :], axis=-1)

    return kappas





    
    