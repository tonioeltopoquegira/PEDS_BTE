import jax.numpy as jnp
import jax.random as jr

import jax.random as jr
from flax.nnx import rnglib, Rngs

def _unwrap_key(key):
    """Turn either
       - a raw jax.PRNGKey   → pass through
       - an nnx.Rngs object  → pull out the first available stream
    """
    # if it isn’t an nnx.Rngs, assume it’s already a PRNGKey
    if not isinstance(key, rnglib.Rngs):
        return key

    # grab the internal dict of streams
    streams = getattr(key, "_rngs", None)
    if not streams:
        raise ValueError("nnx.Rngs contains no RNG streams")

    # pick the first one (you can also inspect streams.keys() if you want a named one)
    return next(iter(streams.values()))


def pcelowrank(mean: jnp.ndarray, var: jnp.ndarray, u: jnp.ndarray, 
               pde_solver, 
               n_collocation: int = 30, key=None):
    
   
    
    collocation_points = low_rank_sample(mean, var, u, n_collocation, key)

    batch_size, n_collocation, _, _ = collocation_points.shape

    flat_points = collocation_points.reshape(-1, 5, 5) 

   

    k_vals = pde_solver(flat_points)  

    k_vals = k_vals.reshape(batch_size, n_collocation)
    
    k_mean = jnp.mean(k_vals, axis=1)
    k_var = jnp.var(k_vals, axis=1)

    return k_mean, k_var


def low_rank_sample(mu: jnp.ndarray, sigma: jnp.ndarray, u: jnp.ndarray, n_samples: int, key):
    batch_size = mu.shape[0]
    d = 25
    
    mu_flat = mu.reshape(batch_size, d)

    if isinstance(key, Rngs):
        # two calls to key() each bump the default-stream counter
        key_epsilon = key()
        key_eta     = key()
    else:
        key_epsilon, key_eta = jr.split(key, 2)

    epsilon = jr.normal(key_epsilon, shape=(batch_size, n_samples, d))
    eta     = jr.normal(key_eta,     shape=(batch_size, n_samples, 1))


    samples_flat = mu_flat[:, None, :] +  sigma[:, None, :]  * epsilon + eta * u[:, None, :] 

    return samples_flat.reshape(batch_size, n_samples, 5, 5)


if __name__ == "__main__":

    from solvers.low_fidelity_solvers.fourier import fourier_solver
    
    key = jr.PRNGKey(44)
    mu = jnp.stack([jnp.ones((5, 5)) * 50.0, jnp.ones((5, 5)) * 30.0])         # (2, 5, 5)
    sigma = jnp.stack([jnp.ones(25) * 10.0, jnp.ones(25) * 5.0])               # (2, 25)
    # Create random low-rank directions u ∈ ℝ^{2 × 25}
    u = jr.normal(key, (2, 25))  # two random u vectors
    u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    u = 20.0 * u  # just an example scale




    k_mean, k_var = pcelowrank(mu, sigma, u, fourier_solver,n_collocation=10, key=key)

    print(k_mean, k_var)