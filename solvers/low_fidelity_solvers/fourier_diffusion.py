import jax
import jax.random as random
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap
from flax import nnx
from jax import custom_vjp
import gc

#from solvers.low_fidelity_solvers.utilities_lowfid import plot_temperature, flux_kappa
from utilities_lowfid import plot_temperature


import matplotlib.pyplot as plt

@nnx.jit
def fourier_solver(diffusivity, iterations=1000, display = False):
    batch_size = diffusivity.shape[0]
    N = diffusivity.shape[1]

    u = jnp.zeros((batch_size, N, N))

    for i in range(N):
        gradient = jnp.linspace(0.5, -0.5, N)
        u = u.at[:, :, i].set(gradient)


    kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
    kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
    kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
    kappa_u = jnp.roll(diffusivity, shift=1, axis=2)

    kappa_sum = kappa_r+kappa_l+kappa_d+kappa_u

    def body_fn(u, _):
        
        u_r = jnp.roll(u, shift=1, axis=1)
        u_l = jnp.roll(u, shift=-1, axis=1)
        u_d = jnp.roll(u, shift=-1, axis=2)
        u_u = jnp.roll(u, shift=1, axis=2)

        
        u_new = update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d, u_r, u_l, u_u, u_d)

        return u_new, None

    # Use lax.scan to run iterations
    Ts, _ = lax.scan(body_fn, u, None, length=iterations)

    return Ts


@nnx.jit
def update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d, u_r, u_l, u_u, u_d):
    # Perform the update step using the shifted arrays
    u_new = ((
        u_r * kappa_r +
        u_l * kappa_l +
        u_d * kappa_d +
        u_u * kappa_u
    ) / kappa_sum)

    # Apply boundary conditions
    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row

    return u_new



if __name__ == "__main__":

    from utilities_lowfid import test_solver
    test_solver(fourier_solver, num_obs=500, jax=True)
