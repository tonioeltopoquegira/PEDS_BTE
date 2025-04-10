import jax
import jax.numpy as jnp
import optax
from models.peds import PEDS
from models.ensembles import ensemble
from models.model_utils import predict


def smoothed_heavside(xi, beta, eta):
    numerator = jnp.tanh(beta * eta) + jnp.tanh(beta * (xi - eta))
    denominator = jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta))
    return numerator / denominator


def gradient_opt(model, target, seed, neigh=True, min_var=False, smoothed=True, batch_size=10, steps=100, lr=0.1):

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
            penalty = 0
            if smoothed:
                # params = smoothed_heavside(params, 2.0, 0.5)
                penalty = jnp.sum(jnp.minimum(jnp.abs(params - 0), jnp.abs(params - 1)))  # L1 distance from 0 or 1

            k, var = predict(model, params)  

            if var is None:
                var = 0.0

            if min_var:
                return jnp.mean((k - target) ** 2) + var + penalty * 0.5
            else:
                return jnp.mean(jnp.abs(k - target)) + penalty * 0.5


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

    # print("k", k.shape)
    # print("target", target)
    # print("params", params.shape)
    # print("best_params", best_params.shape)
    # print("best_k_binarized", best_k_binarized.shape)

    print(f"Best Found params: {best_params} with kappa= {best_k}")
    print(f"Binarized {best_binary_params} with kappa={best_k_binarized}")

    processed_params = smoothed_heavside(best_params.reshape((1, 25)), 2.0, 0.5)
    binary_params = (processed_params > 0.5).astype(jnp.float32)

    # print("processed_params", processed_params.shape)

    k, _ = predict(model, processed_params)  # Model output
    k_binarized, _ = predict(model, binary_params)  # Model output
    print("----------------")
    print(f"Best Found params: {processed_params} with kappa= {k}")
    print(f"Binarized {binary_params} with kappa={k_binarized}")


    return best_binary_params, best_k_binarized


