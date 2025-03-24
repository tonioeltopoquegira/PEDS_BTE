import jax
import jax.numpy as jnp
import optax

def gradient_opt(model, target, out, seed, neigh=True, batch_size=10, steps=100, lr=0.1):

    """
    Model takes binary array of size 25 and outputs a float k
    I want to find the binary array that gives me the k closest to target. Ignore out.
    I want to use a gradient optimization, potentially using also momentum (cheap approximation of hessian).
    My code is very fast to compute batch of values, basically same time as computing 1... can we exploit this? I am also happy to use library
    Model is jax. Binary to continuous relaxation needed?
    """
    if neigh:
        def loss_fn(params, model, target):
            noise = jax.random.normal(jax.random.PRNGKey(0), params.shape) * 0.05
            perturbed_params = jnp.clip(params + noise, 0, 1)  

            if out == "peds":
                k, _ = model(perturbed_params)
            else:
                k = model(perturbed_params)
        
            return jnp.mean(jnp.abs(k - target))
    
    else:
        pass

    seed = seed.unwrap() if hasattr(seed, "unwrap") else seed # Extract JAX key if it's an nnx RngStream

    params = jax.random.uniform(seed, (batch_size, 25))  # Continuous relaxation in [0,1]

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

    if out=="peds":
            k, _ = model(params)  # Model output
            k_binarized, _ = model(binary_params)
    else:
        k = model(params)

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



