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
from models.cnn import cnn
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid

# Define Parallelization
devices = jax.devices()

#mesh = Mesh(devices=np.array(jax.devices()), axis_names=('batch',))

n_devices = jax.device_count()
print(n_devices)

# model name for params and figures
path = "base_peds"

# Ingest data <- Here we will do active learning
full_data = jnp.load("data/highfidelity/high_fidelity_10012_20steps.npz", allow_pickle=True)

pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
base_conductivities = jnp.asarray(full_data['conductivity'], dtype=jnp.float32)

# Create dataset
dataset_train = [pores[:1500], base_conductivities[:1500], kappas[:1500]]
dataset_valid = [pores[1500:1750], base_conductivities[1500:1750], kappas[1500:1750]]

# Create model
key = nnx.Rngs(42)
generator = mlp(input_size= 25, hidden_sizes=[32, 64, 128], step_size=5, rngs=key)
#generator = cnn(rngs=key)

# Params initializing or restoring
generator, checkpointer, ckpt_dir = initialize_or_restore_params(generator, model_name='peds_PI')

# Low Fidelity Solver
lowfidsolver = lowfid(solver='gauss', iterations=1000)

# Train the model
train_model(model_name = 'peds_PI',
        dataset_train=dataset_train, dataset_valid = dataset_valid,
        generator=generator, 
        lowfidsolver=lowfidsolver, 
        learn_rate_min= 5e-5, learn_rate_max=5e-4, schedule='cosine-cycles', epochs=2000, batch_size = 100,
        checkpointer = checkpointer, ckpt_dir=ckpt_dir
        )

