import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np

from models.peds import PEDS
from modules.params_utils import update_params
from modules.training import train_model


from models.mlp import mlp
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid


# Define Parallelization
# Create dataset
mesh = Mesh(devices=np.array(jax.devices()),
            axis_names=('batch',))

data_sharding = NamedSharding(mesh, PartitionSpec('batch',))

# model name for params and figures
path = "base_peds"

# Ingest data <- Here we will do active learning
full_data = jnp.load("data/highfidelity/high_fidelity_10012_100steps.npz", allow_pickle=True)

pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)[:10000]
kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)[:10000]
base_conductivities = jnp.asarray(full_data['conductivity'], dtype=jnp.float32)[:10000]

# Create dataset


dataset = [pores, base_conductivities, kappas]

n_devices = jax.device_count()
print(n_devices)

# Shard dataset
dataset_sharded = (
    pores.reshape(pores.shape[0] // n_devices, n_devices, 25),
    base_conductivities.reshape(pores.shape[0] // n_devices, n_devices, 100, 100),
    kappas.reshape(pores.shape[0] // n_devices, n_devices),
)


# Create model
key = nnx.Rngs(42)

generator = mlp(input_size= 25, hidden_sizes=[5], step_size=1, rngs=key)
lowfidsolver = lowfid(iterations=5000)


train_model(dataset=dataset_sharded, epochs=1, generator=generator, lowfidsolver=lowfidsolver)

