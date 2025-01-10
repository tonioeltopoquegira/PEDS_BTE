import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from modules.training_utils import data_loader, print_generated, update_and_check_grads, clip_gradients, plot_learning_curves, choose_schedule

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
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid

from mpi4py import MPI


# Data distribution among MPI ranks
def distribute_dataset(dataset, rank, size):
    """
    Distribute dataset among MPI ranks.

    Args:
    - dataset: List of arrays [pores, conductivities, kappas].
    - rank: Current MPI rank.
    - size: Total number of MPI processes.

    Returns:
    - Local dataset for this rank.
    """
    pores, conductivities, kappas = dataset

    # Determine chunk size per rank
    n_samples = pores.shape[0]
    chunk_size = n_samples // size

    assert n_samples % size == 0, "Dataset size must be divisible by the number of MPI processes"

    # Compute local data slice
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size

    # Slice the dataset for this rank
    local_pores = pores[start_idx:end_idx]
    local_conductivities = conductivities[start_idx:end_idx]
    local_kappas = kappas[start_idx:end_idx]

    return [local_pores, local_conductivities, local_kappas]



def predict(generator, lowfidsolver, pores, conductivities):

    conductivity_res = nnx.jit(generator)(pores)
        
    new_conductivity = conductivity_res+conductivities 

    new_conductivity = jnp.maximum(new_conductivity, 1e-5) # here we 
    
    kappa = lowfidsolver(new_conductivity) 
    
    return kappa, conductivity_res

def mpi_allreduce_gradients(local_grads):
    # Perform MPI Allreduce to accumulate gradients across all ranks
    return jax.tree_util.tree_map(
        lambda x: comm.allreduce(x, op=MPI.SUM), local_grads
    )

# Define local data loader
def data_loader(*arrays, batch_size):
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield tuple(array[batch_indices] for array in arrays)

def train_step(generator, lowfidsolver, pores, conductivities, kappas):
    def loss_fn(generator):
        kappa_pred, conductivity_res = predict(generator, lowfidsolver, pores, conductivities)
        residuals = kappa_pred - kappas
        return jnp.sum(residuals**2)

    loss, grads = nnx.value_and_grad(loss_fn)(generator)
    return loss, grads

def accumulate_gradients(total_grads, new_grads):
        if total_grads is None:
            return new_grads
        return jax.tree_util.tree_map(lambda x, y: x + y, total_grads, new_grads)


# Create model
key = nnx.Rngs(42)
generator = mlp(input_size= 25, hidden_sizes=[32, 64, 128], step_size=5, rngs=key)

# Params initializing or restoring
generator, checkpointer, ckpt_dir = initialize_or_restore_params(generator, model_name='peds_PI')

# Low Fidelity Solver
lowfidsolver = lowfid(solver='direct', iterations=1000)


schedule = "constant"
learn_rate_min = 5e-5
learn_rate_max = 5e-4
batch_size = 200
epochs = 20

lr_schedule = choose_schedule(schedule, learn_rate_min, learn_rate_max, epochs)
optimizer = nnx.Optimizer(generator, optax.adam(lr_schedule))

print("Training")

import sys
sys.stdout.flush()

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Current process ID
size = comm.Get_size()  # Total number of processes

#print(f"Process {rank} of {size} initialized")

# Ingest data <- Here we will do active learning
full_data = jnp.load("data/highfidelity/high_fidelity_10012_20steps.npz", allow_pickle=True)

pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
base_conductivities = jnp.asarray(full_data['conductivity'], dtype=jnp.float32)

# Create dataset
dataset_train = [pores[:8000], base_conductivities[:8000], kappas[:8000]]
dataset_valid = [pores[8000:], base_conductivities[8000:], kappas[8000:]]

# Distribute the training and validation datasets
dataset_train_local = distribute_dataset(dataset_train, rank, size)
dataset_valid_local = distribute_dataset(dataset_valid, rank, size)

print(dataset_train_local[0].shape)


for epoch in range(epochs):
    epoch_time = time.time()

    grads = None  # Initialize accumulated gradients
    total_loss = 0.0  # Initialize total loss for the epoch

    for en, batch in enumerate(data_loader(*dataset_train_local, batch_size=batch_size)):
        pores_local, conductivities_local, kappas_local = batch
        
        # Compute loss and gradients locally
        local_loss, local_grads = train_step(generator, lowfidsolver, pores_local, conductivities_local, kappas_local)

        print(f"Batch {en} done for rank {rank}")
        
        # Accumulate loss across ranks
        total_loss += comm.allreduce(local_loss, op=MPI.SUM)

        grads = accumulate_gradients(grads, mpi_allreduce_gradients(local_grads))

    

    # Compute average loss across the entire dataset
    avg_loss = total_loss / dataset_train[0].shape[0]

    sys.stdout.flush()
    if rank == 0:
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Epoch Time: {time.time() - epoch_time:.2f}s")

    

    # Update model parameters using globally accumulated gradients
    optimizer.update(grads)


