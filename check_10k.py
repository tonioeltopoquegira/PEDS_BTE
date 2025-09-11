import jax.numpy as jnp

# Load data
full_data = jnp.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)

# Flatten kappas if needed
kappas = kappas.flatten()

# Sort kappas and get indices
sorted_indices = jnp.argsort(kappas)
sorted_kappas = kappas[sorted_indices]

# Pick 100 equidistant indices from the sorted kappas
n_samples = 100
total = len(kappas)
step = total // n_samples
selected_sorted_indices = jnp.arange(0, step * n_samples, step)

# Get the original indices corresponding to these sorted ones
selected_indices = sorted_indices[selected_sorted_indices]

# Select the corresponding pores and kappas
selected_pores = pores[selected_indices]
selected_kappas = kappas[selected_indices]
