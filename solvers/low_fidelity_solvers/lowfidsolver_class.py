from flax import nnx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax


from solvers.low_fidelity_solvers.gauss_diffusion import gauss_solver
from solvers.low_fidelity_solvers.fd_diffusion import fd_diffusion
#from solvers.low_fidelity_solvers.fourier import fourier_solver

class lowfid(nnx.Module):
   
    def __init__(self, solver="gauss", iterations=5000):
        self.iterations = iterations
        self.solver = solver

    #@nnx.vmap(in_axes=0, out_axes=0)
    def __call__(self, conductivity):
        if self.solver == "gauss":
            T = gauss_solver(conductivity)
            kappas = flux_kappa(conductivity, T)
        if self.solver == "direct":
            T = fd_diffusion(conductivity)
            kappas = flux_kappa(conductivity, T)
        
        #if self.solver == "fourier":
        #    T, kappas = fourier_solver(conductivity)
            
        return kappas
    


#@nnx.jit
def flux_kappa(conductivity, Ts):
    # Initialize arrays for fluxes
    Jy = jnp.zeros_like(Ts)
    
    Jy = -conductivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / 1.0
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    kappas = jnp.sum(Jy[:,   conductivity.shape[1] // 2, :], axis=-1)

    return kappas





    
    