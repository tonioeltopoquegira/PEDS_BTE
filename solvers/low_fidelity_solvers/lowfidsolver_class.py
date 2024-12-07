from flax import nnx
import jax.numpy as jnp
import numpy as np
from jax import lax
import matplotlib.pyplot as plt
import jax

from fourier_diffusion import fourier_solver

class lowfid(nnx.Module):
   
    def __init__(self, iterations=5000):
        self.iterations = iterations

    @nnx.vmap(in_axes=0, out_axes=0)
    def __call__(self, diffusivity):
        T = fourier_solver(diffusivity)
        return flux_kappa(diffusivity, T)

@nnx.jit
def flux_kappa(diffusivity, Ts):
    # Initialize arrays for fluxes
    Jy = jnp.zeros_like(Ts)
    
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / 1.0
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    kappas = jnp.sum(Jy[:,   diffusivity.shape[1] // 2, :], axis=-1)

    return kappas





    
    