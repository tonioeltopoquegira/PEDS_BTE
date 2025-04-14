import jax
import jax.numpy as jnp
import optax
from models.peds import PEDS
from models.ensembles import ensemble
from models.model_utils import predict

def gradient_opt(model, target, stochastic, seed, var_param=1.00, steps=100, lr=0.1, debug=True):

    
        
    def loss_fn(params, model, target):
        k, var = predict(model, params) 
        if stochastic:
            print(f"Difference {jnp.abs(k-target).item()}, Variance {var.item()}, Total {jnp.abs(k-target).item() + var_param * var.item()}")
            return jnp.abs(k - target) + var_param * var
        else:
            return jnp.abs(k - target)


    seed = seed.unwrap() if hasattr(seed, "unwrap") else seed # Extract JAX key if it's an nnx RngStream

    params = jax.random.uniform(seed, (1, 25))  # Continuous relaxation

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

    k, var = predict(model, params)  # Model output
    k_binarized, var = predict(model, binary_params)  # Model output

    print(f"Best Found params: {params} with kappa= {k.item()}")
    print(f"Binarized {binary_params} with kappa={k_binarized}")

    return binary_params, k, var


