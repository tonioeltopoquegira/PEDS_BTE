import os
from mpi4py import MPI
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  

from flax import nnx
import numpy as np

from modules.data_ingestion import data_ingestion
from modules.choose_model import select_model
from modules.params_utils import initialize_or_restore_params
from modules.training import train_model


# Experiment configuration
exp = { "seed" : 42,

        # Data
        "filename_data" : "high_fidelity_2_13000.npz",
        "train_size" : 8000,
        "total_size" : 10000,
        "stratified": False,
        "multifidelity": False,

        # Model
        "model_name": "PEDS_gauss_5by5_lowval_100kparams",
        "model": "PEDS",
        "resolution": 5,
        "learn_residual": False,
        "hidden_sizes": [32, 64, 128, 256, 256],
        "activation": "relu",
        "solver": "gauss",
        "init_min": 1e-12,

        # Training
        "epochs": 3000,
        "batch_size": 200,
        "learn_rate_max": 5e-4,
        "learn_rate_min": 5e-5,
        "schedule": "cosine-cycles",
        "print_every": 10,

        # Results
        "mse_train": -1.0,
        "mse_test": -1.0,
        "perc_error":-1.0
    }

# Initialize random key
rngs = nnx.Rngs(exp["seed"])

# Ingest data
dataset_train, dataset_valid = data_ingestion(
    filename=exp["filename_data"], 
    train_size=exp["train_size"], 
    total_size=exp['total_size'],
    stratified=exp["stratified"], 
    multifidelity=exp["multifidelity"], 
    key=rngs
)

model = select_model(
    rngs=rngs, 
    model_type=exp["model"], 
    resolution=exp["resolution"], 
    learn_residual=exp["learn_residual"], 
    hidden_sizes=exp["hidden_sizes"], 
    activation=exp["activation"], 
    solver=exp["solver"],
    init_min = exp['init_min']
)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Current process ID
size = comm.Get_size()  # Total number of processes

if rank == 0:
    print(f"Training on {size} devices...")

# Params initializing or restoring
model, checkpointer = initialize_or_restore_params(model, exp["model_name"], rank=rank)

# Train model
train_model(
    exp=exp, 
    model_name=exp["model_name"],
    dataset_train=dataset_train,
    dataset_valid=dataset_valid,
    model=model,
    learn_rate_min=exp["learn_rate_min"], 
    learn_rate_max=exp["learn_rate_max"], 
    schedule=exp["schedule"], 
    epochs=exp["epochs"], 
    batch_size=exp["batch_size"],
    checkpointer=checkpointer,
    print_every=exp["print_every"]
)
