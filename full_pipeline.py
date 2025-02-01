import os
from mpi4py import MPI
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  

import jax.numpy as jnp
from flax import nnx

#from modules.data_ingestion import ingest
from modules.params_utils import initialize_or_restore_params
from modules.training import train_model

from models.mlp import mlp
from models.peds import PEDS
import shutil


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Current process ID
size = comm.Get_size()  # Total number of processes

if rank == 0:
        print(f"Training on {size} devices...")

# Ingest data <- Here we will do active learning
full_data = jnp.load("data/highfidelity/high_fidelity_10012_20steps.npz", allow_pickle=True)

pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
base_conductivities = jnp.asarray(full_data['conductivity'], dtype=jnp.float32)

# take kappas, divide it in percentiles of 10th, stratify sample (same number) 

# Create dataset
dataset_train = [pores[:8000], base_conductivities[:8000], kappas[:8000]]
dataset_valid = [pores[8000:], base_conductivities[8000:], kappas[8000:]]
        
# Model creation and name (experiment name)
model_name = "PEDS_fourier"
model = PEDS(resolution = 20, learn_residual= False, hidden_sizes= [32, 64], activation="relu", solver="gauss") # parameters: 60k
#rngs = nnx.Rngs(42)
#model = mlp(layer_sizes=[25, 32, 64, 128, 128, 256, 1], activation="relu", rngs=rngs) # 

# Params initializing or restoring
model, checkpointer = initialize_or_restore_params(model, model_name, rank=rank)

train_model(
    model_name=model_name,
    dataset_train=dataset_train,
    dataset_valid=dataset_valid,
    model=model,
    learn_rate_min=5e-5, learn_rate_max=5e-4, schedule='cosine-cycles', 
    epochs=3000, batch_size=250,
    checkpointer=checkpointer,
    print_every=10
)




