import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  


import jax
from jax import debug, sharding
from flax import nnx

import jax.numpy as jnp
import numpy as np
import time

from modules.training_utils import data_loader
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Ingest data <- Here we will do active learning
full_data = jnp.load("data/highfidelity/high_fidelity_10012.npz", allow_pickle=True)

pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)[:10000]
kappas = jnp.asarray(full_data['kappa_bte'], dtype=jnp.float32)[:10000]
pores = jnp.tile(pores, (10, 1))  # Repeat along the first axis
kappas = jnp.tile(kappas, 10)  # Repeat along the first axis



print(f"Pores shape: {pores.shape}")
print(f"Kappas shape: {kappas.shape}")


# Create dataset
mesh = Mesh(devices=np.array(jax.devices()),
            axis_names=('batch',))

data_sharding = NamedSharding(mesh, PartitionSpec('batch',))


def func(arr1):
    return jnp.sum(arr1**2)

# Parallelized function
@jax.pmap
def func_parallel(arr1):
    return jnp.sum(arr1**2, axis=1)  # Ensure batch axis is reduced correctly

# Simulated dataset and data loader
def data_loader(arr1, arr2, batch_size):
    # Simulates yielding batches from dataset
    num_batches = len(arr1) // batch_size
    for i in range(num_batches):
        yield arr1[i * batch_size : (i + 1) * batch_size], arr2[i * batch_size : (i + 1) * batch_size]

# Simulated data for demonstration
n_devices = jax.device_count()
print(n_devices)

# Shard dataset
dataset_sharded = (
    pores.reshape(n_devices, -1, pores.shape[-1]),
    kappas.reshape(n_devices, -1, pores.shape[-1]),
)

# Non-parallelized version
def bigfun(dataset, batch_size):
    sum_tot = 0
    for batch in data_loader(*dataset, batch_size=batch_size):
        arr1_no_shard, arr2_no_shard = batch
        #print(f"Arra1 shape:, {arr1_no_shard.shape}")
        sum_tot += func(arr1_no_shard)
    return sum_tot

# Parallelized version
def bigfun_parallel(dataset_sharded, batch_size):
    sum_tot = 0
    for batch in data_loader(*dataset_sharded, batch_size=batch_size // n_devices):
        arr1_shard, arr2_shard = batch
        print(f"Arra1 shape:, {arr1_shard.shape}")
        #print(f"arr1 Shard: {jax.debug.visualize_array_sharding(arr1_shard)}")
        sum_tot += jnp.sum(func_parallel(arr1_shard))
    return sum_tot

# Example batch size
batch_size = 10

# Timing non-parallelized function
time_no_sharding = time.time()
res = bigfun((pores, kappas), batch_size)
print(f"Time (no sharding): {time.time() - time_no_sharding:.4f}, Result: {res}")

# Timing parallelized function
time_sharding = time.time()
res = bigfun_parallel(dataset_sharded, batch_size)
print(f"Time (sharding): {time.time() - time_sharding:.4f}, Result: {res}")