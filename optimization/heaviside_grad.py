import time
import jax
import jax.numpy as jnp
import optax
from models.model_utils import predict


def smoothed_heavside(xi, beta, eta):
    numerator = jnp.tanh(beta * eta) + jnp.tanh(beta * (xi - eta))
    denominator = jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta))
    return numerator / denominator

def loss_fn(params, model, target, beta, use_smoothed=True, min_var=False):
        if use_smoothed:
            params_used = smoothed_heavside(params, beta, 0.5)
        else:
            params_used = params

        k, var = predict(model, params_used)
        if var is None:
            var = 0.0

        if min_var:
            loss = (k - target) ** 2 + var 
        else:
            loss = jnp.abs(k - target) 

        return loss

def gradient_opt_heaviside(model, target, seed, min_var=False, beta_start=25.0, beta_increase=1.0125, lr=0.01, beta_int=1, use_smoothed=True, use_penalty=False, batch_size=1, steps=1000):

    seed = seed.unwrap() if hasattr(seed, "unwrap") else seed
    #params = jax.random.uniform(seed, (batch_size, 25))  # Initial continuous parameters

   
    mask = jax.random.bernoulli(seed, p=0.5, shape=(batch_size, 25))
    params = jnp.where(mask, 0.6, 0.4)


    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    beta = beta_start
    loss_history = []
    convergence_threshold = 5e-3
    binary_tolerance = 0.05
    patience = 10
    step_time = 0.0

    step_count = 0

    for step_idx in range(steps):

        # ===== Evaluate BEFORE update =====
        params_smoothed = smoothed_heavside(params, beta, 0.5)
        binary_params = (params_smoothed > 0.5).astype(jnp.float32)
        k, var = predict(model, params)
        k_smoothed, _ = predict(model, params_smoothed)
        k_binary, _ = predict(model, binary_params)

        if step_idx % 1 == 0:
            print(f"[Step {step_idx-1}] [Beta {beta}] K (original, smoothed, binary): , {k:.4f}, {k_smoothed:.4f}, {k_binary:.2f}, [Time] {step_time:.2f} seconds")
        
        step_start = time.time()  # Start timing the step

        if step_idx % 15 == 0:
            jax.clear_caches()

        loss, grads = jax.value_and_grad(loss_fn)(params, model, target, beta, use_smoothed, min_var)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        loss_history.append(loss)
        if len(loss_history) > patience:
            loss_history.pop(0)

        # Check if loss has converged (stabilized)
        loss_converged = (
            len(loss_history) == patience and
            jnp.max(jnp.abs(jnp.diff(jnp.array(loss_history)))) < convergence_threshold
        )

        # Check if parameters are near-binary
        avg_deviation = jnp.mean(jnp.minimum(params_smoothed, 1 - params_smoothed))
        binary_converged = avg_deviation < binary_tolerance

        # === Early beta update if loss has converged, even if not binary ===
        if loss_converged:
            print(f"[Loss converged at step {step_idx}] → Increasing beta from {beta:.2f} to {beta * beta_increase:.2f}")
            beta *= beta_increase
            step_count = 0
            loss_history = []  # reset history
            continue  # move to next beta

        """if (step_count + 1) == beta_int:
            beta *= beta_increase
            step_count = 0"""
        
        step_count += 1
        
        step_time = time.time() - step_start  # End timing

            #print(f"[Beta Update] Increased to {beta:.2f}")

    # Final evaluation
    binary_params = (params > 0.5).astype(jnp.float32)
    k, _ = predict(model, params)
    k_binarized, _ = predict(model, binary_params)
    losses = jnp.abs(k - target)
    best_idx = jnp.argmin(losses)

    print("========== FINAL RESULT ==========")
    print(f"Best continuous params:\n{params}")
    print(f"→ kappa: {k}")
    print(f"Best binary params:\n{binary_params}")
    print(f"→ binarized kappa: {k_binarized}")
    print("==================================")

    processed_params = smoothed_heavside(params.reshape((1, 25)), beta, 0.5)
    binary_params_final = (processed_params > 0.5).astype(jnp.float32)
    k_final, _ = predict(model, processed_params)
    k_bin_final, var_k = predict(model, binary_params_final)

    print("\nRefined outputs:")
    print(f"Processed params (smoothed): {processed_params}")
    print(f"Binarized params: {binary_params_final}")
    print(f"kappa (smoothed): {k_final}, kappa (binarized): {k_bin_final}")

    return binary_params, k_bin_final, var_k 
