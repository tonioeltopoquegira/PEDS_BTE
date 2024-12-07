import jax.numpy as jnp

def data_loader(*arrays, batch_size):
    """
    A simple data loader for batching data arrays.

    Args:
        arrays: One or more JAX arrays with the same first dimension.
        batch_size: The number of samples per batch.

    Yields:
        Batches of data arrays.
    """
    # Ensure all arrays have the same number of samples
    n_samples = arrays[0].shape[0]
    for array in arrays:
        assert array.shape[0] == n_samples, "All input arrays must have the same first dimension."
    
    indices = jnp.arange(n_samples)  # Use jnp.arange for JAX arrays
    
    # Split into batches and yield
    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(array[batch_indices] for array in arrays)