import jax.numpy as jnp
import jax.random as jr

def pcelowrank(mean: jnp.ndarray, var: jnp.ndarray, u: jnp.ndarray, 
               pde_solver, 
               n_collocation: int = 30, key=None):
    
    collocation_points = low_rank_sample(mean, var, u, n_collocation, key)

    batch_size, n_collocation, _, _ = collocation_points.shape

    flat_points = collocation_points.reshape(-1, 5, 5)  

    T_vals, k_vals = pde_solver(flat_points)  

    k_vals = k_vals.reshape(batch_size, n_collocation)
    
    k_mean = jnp.mean(k_vals, axis=1)
    k_var = jnp.var(k_vals, axis=1)

    return k_mean, k_var


def low_rank_sample(mu: jnp.ndarray, sigma: jnp.ndarray, u: jnp.ndarray, n_samples: int, key):
    batch_size = mu.shape[0]
    d = 25
    
    mu_flat = mu.reshape(batch_size, d)

    key_epsilon, key_eta = jr.split(key)
    epsilon, eta = jr.normal(key_epsilon, shape=(batch_size, n_samples, d)) , jr.normal(key_eta, shape=(batch_size, n_samples, 1))     

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