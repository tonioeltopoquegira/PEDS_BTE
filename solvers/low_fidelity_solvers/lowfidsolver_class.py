from flax import nnx
import jax.numpy as jnp

from solvers.low_fidelity_solvers.fourier_diffusion import fourier_solver
from solvers.low_fidelity_solvers.fd_diffusion import fd_diffusion

class lowfid(nnx.Module):
   
    def __init__(self, solver="gauss", iterations=5000):
        self.iterations = iterations
        self.solver = solver

    #@nnx.vmap(in_axes=0, out_axes=0)
    def __call__(self, conductivity):
        if self.solver == "gauss":
            T = fourier_solver(conductivity)
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





    
    