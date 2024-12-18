import os
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np

from models.peds import PEDS
from modules.params_utils import initialize_or_restore_params
from modules.training import train_model

from models.mlp import mlp
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid


# Define Parallelization
devices = jax.devices()

mesh = Mesh(devices=np.array(jax.devices()), axis_names=('batch',))

n_devices = jax.device_count()
print(n_devices)

# model name for params and figures
path = "base_peds"

# Ingest data <- Here we will do active learning
full_data = jnp.load("data/highfidelity/high_fidelity_10012_20steps.npz", allow_pickle=True)

pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)[:100]
kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)[:100]
base_conductivities = jnp.asarray(full_data['conductivity'], dtype=jnp.float32)[:100]

# Create dataset

"""data_sharding = NamedSharding(mesh, PartitionSpec('batch',))
pores = jax.device_put(pores, data_sharding)
kappas = jax.device_put(kappas, data_sharding)"""
dataset = [pores, base_conductivities, kappas]


# Create model
key = nnx.Rngs(42)

generator = mlp(input_size= 25, hidden_sizes=[64, 256], step_size=5, rngs=key)

# Params initializing or restoring

generator, checkpointer, ckpt_dir = initialize_or_restore_params(generator, model_name='base_peds')


lowfidsolver = lowfid(solver='direct', iterations=1000)



train_model(dataset=dataset, epochs=100, generator=generator, lowfidsolver=lowfidsolver, checkpointer = checkpointer, ckpt_dir=ckpt_dir)

