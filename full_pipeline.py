import os
from mpi4py import MPI
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  

import jax.numpy as jnp
from flax import nnx

from modules.params_utils import initialize_or_restore_params
from modules.training import train_model

from models.mlp import mlp
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Current process ID
size = comm.Get_size()  # Total number of processes

# Ingest data <- Here we will do active learning
full_data = jnp.load("data/highfidelity/high_fidelity_10012_20steps.npz", allow_pickle=True)

pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
base_conductivities = jnp.asarray(full_data['conductivity'], dtype=jnp.float32)

# Create dataset
dataset_train = [pores[:8000], base_conductivities[:8000], kappas[:8000]]
dataset_valid = [pores[8000:], base_conductivities[8000:], kappas[8000:]]

# Create model
key = nnx.Rngs(42)
generator = mlp(input_size= 25, hidden_sizes=[32, 64, 128], step_size=5, rngs=key)

# Params initializing or restoring
generator, checkpointer, ckpt_dir = initialize_or_restore_params(generator, model_name='peds_PI', rank=rank)

# Low Fidelity Solver
lowfidsolver = lowfid(solver='gauss', iterations=1000)

train_model(
    model_name='peds_PI',
    dataset_train=dataset_train,
    dataset_valid=dataset_valid,
    generator=generator, 
    lowfidsolver=lowfidsolver, 
    physics_bias= 'geom-cap', # "loss", 
    learn_rate_min=5e-5, learn_rate_max=5e-4, schedule='cosine-cycles', 
    epochs=3000, batch_size=200,
    checkpointer=checkpointer
)






