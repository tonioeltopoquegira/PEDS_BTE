import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import device_put
import numpy as np

from models.peds import PEDS
from modules.params_utils import initialize_or_restore_params
from modules.training import train_model

from models.mlp import mlp
from models.cnn import cnn
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid

# Define Parallelization
n_devices = len(jax.devices())

mesh = Mesh(devices=np.array(jax.devices()), axis_names=('devices',))
data_sharding = NamedSharding(mesh, PartitionSpec('devices',))

print(n_devices)

# model name for params and figures


# Ingest data <- Here we will do active learning
full_data = jnp.load("data/highfidelity/high_fidelity_10012_20steps.npz", allow_pickle=True)

pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
base_conductivities = jnp.asarray(full_data['conductivity'], dtype=jnp.float32)

# Create dataset
dataset_train = [pores[:8000], base_conductivities[:8000], kappas[:8000]]
dataset_valid = [pores[8000:], base_conductivities[8000:], kappas[8000:]]

# Shard the dataset
def shard_dataset(dataset, n_device, sharding):
    # Extract components of the dataset
    pores, conductivities, kappas = dataset
    
    # Determine shard sizes
    shard_size = pores.shape[0] // n_devices
    
    # Ensure each shard is of equal size
    assert pores.shape[0] % n_devices == 0, "Dataset size must be divisible by the number of devices"
    
    # Reshape data to distribute across devices
    pores_sharded = pores.reshape(n_devices, shard_size, *pores.shape[1:])
    conductivities_sharded = conductivities.reshape(n_devices, shard_size, *conductivities.shape[1:])
    kappas_sharded = kappas.reshape(n_devices, shard_size, *kappas.shape[1:])

    # Apply NamedSharding to the reshaped data
    pores_sharded = device_put(pores_sharded, sharding)
    conductivities_sharded = device_put(conductivities_sharded, sharding)
    kappas_sharded = device_put(kappas_sharded, sharding)

    return (pores_sharded, conductivities_sharded, kappas_sharded)


# Shard training and validation datasets
#dataset_train_sharded = shard_dataset(dataset_train, n_devices, data_sharding)
#dataset_valid_sharded = shard_dataset(dataset_valid, n_devices, data_sharding)

# Create model
key = nnx.Rngs(42)
generator = mlp(input_size= 25, hidden_sizes=[32, 64, 128], step_size=5, rngs=key)
#generator = cnn(rngs=key)

# Params initializing or restoring
generator, checkpointer, ckpt_dir = initialize_or_restore_params(generator, model_name='peds_PI')

# Low Fidelity Solver
lowfidsolver = lowfid(solver='gauss', iterations=1000)

train_model(
    model_name='peds_PI',
    dataset_train=dataset_train,
    dataset_valid=dataset_valid,
    generator=generator, 
    lowfidsolver=lowfidsolver, 
    learn_rate_min=5e-5, learn_rate_max=5e-4, schedule='constant', 
    epochs=20, batch_size=200,
    checkpointer=checkpointer
)






