import jax
import jax.numpy as jnp
import optax
from models.peds import PEDS
from models.ensembles import ensemble
from models.model_utils import predict

def gradient_opt(model, target, seed, neigh=True, min_var=False, batch_size=10, steps=100, lr=0.1):

    if neigh:

        def loss_fn(params, model, target):
            noise = jax.random.normal(jax.random.PRNGKey(0), params.shape) * 0.05
            perturbed_params = jnp.clip(params + noise, 0, 1)  

            k, var = predict(model, perturbed_params)  

            if var is None:
                var = 0.0
                
            if min_var:
                return jnp.mean((k - target) ** 2) + var
            else:
                return jnp.mean(jnp.abs(k - target))
    
    else:
        
        def loss_fn(params, model, target):
            k, var = predict(model, params)  

            if var is None:
                var = 0.0

            if min_var:
                return jnp.mean((k - target) ** 2) + var
            else:
                return jnp.mean(jnp.abs(k - target))


    seed = seed.unwrap() if hasattr(seed, "unwrap") else seed # Extract JAX key if it's an nnx RngStream

    params = jax.random.uniform(seed, (batch_size, 25))  # Continuous relaxation

    # Optimizer with momentum
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)


    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, target)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Optimization loop
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state)
        if _ % 25 == 0:
            print(f"Step {_} losses mean: {loss}")
    
    # Binarization step
    binary_params = (params > 0.5).astype(jnp.float32)

    k, _ = predict(model, params)  # Model output
    k_binarized, _ = predict(model, binary_params)  # Model output

    # Compute loss for each element in batch
    losses = jnp.abs(k - target)  # Loss for continuous params
    best_idx = jnp.argmin(losses)  # Index of best parameter set

    # Select best parameters
    best_params = params[best_idx]
    best_binary_params = binary_params[best_idx]
    best_k = k[best_idx]
    best_k_binarized = k_binarized[best_idx]

    print(f"Best Found params: {best_params} with kappa= {best_k}")
    print(f"Binarized {best_binary_params} with kappa={best_k_binarized}")

    return best_binary_params, best_k_binarized


