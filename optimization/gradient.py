import jax
import jax.numpy as jnp
import optax
from models.peds import PEDS
from models.ensembles import ensemble
from models.model_utils import predict

def gradient_opt(model, target, stochastic, seed, var_param=1.00, steps=100, lr=0.1, debug=True, tol=1e-5):
    
    def loss_fn(params, model, target):
        k, var = predict(model, params) 
        if stochastic:
            if debug:
                print(f"Difference {jnp.abs(k - target).item()}, Variance {var.item()}, Total {jnp.abs(k - target).item() + var_param * var.item()}")
            return jnp.abs(k - target) + var_param * var
        else:
            return jnp.abs(k - target)

    # Ensure seed is a JAX PRNGKey
    seed = seed.unwrap() if hasattr(seed, "unwrap") else seed
    params = jax.random.uniform(seed, (1, 25))  # Initial guess

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, target)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    prev_loss = None
    for i in range(steps):
        params, opt_state, loss = step(params, opt_state)
        loss_val = loss.item()

        if debug and i % 25 == 0:
            print(f"Step {i}, loss: {loss_val}")

        if prev_loss is not None and abs(prev_loss - loss_val) < tol:
            if debug:
                print(f"Converged at step {i} with loss change {abs(prev_loss - loss_val)} < tol={tol}")
            break

        prev_loss = loss_val

    # Binarization step
    binary_params = (params > 0.5).astype(jnp.float32)

    k, var = predict(model, params)
    k_binarized, _ = predict(model, binary_params)

    print(f"Best Found params: {params} with kappa = {k.item()}")
    print(f"Binarized params: {binary_params} with kappa = {k_binarized}")

    return binary_params, k, var
