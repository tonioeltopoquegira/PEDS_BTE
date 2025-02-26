import os
from mpi4py import MPI
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  

from flax import nnx

from modules.data_ingestion import data_ingestion
from modules.choose_model import select_model
from modules.params_utils import initialize_or_restore_params
from modules.training import train_model
from modules.training_utils import create_folders

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Current process ID
size = comm.Get_size()  # Total number of processes

# Experiment configuration
exp = { "seed" : 42,

        # Data
        "filename_data" : "high_fidelity_2_13000.npz",
        "train_size" : 12000,
        "total_size" : 13000,
        "stratified": "big->small",

        # Model
        "model_name": "generalization/PEDSres_gauss5_bigsmall",
        "model": "PEDS",
        "resolution": 5,
        "learn_residual": True,
        "hidden_sizes":[64, 256, 256],
        "activation": "relu", #"relu", "hardtanh", "mixed"
        "solver": "gauss",
        "init_min": 1e-16,
        "initialization": "he",
        "final_init": False,

        # Training
        "epochs": 3000,
        "batch_size": 240,
        "learn_rate_max": 1e-3,
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

create_folders(exp['model_name'])

# Ingest data
dataset_train, dataset_valid = data_ingestion(
    rank = rank,
    model_name = exp['model_name'],
    filename=exp["filename_data"], 
    train_size=exp["train_size"], 
    total_size=exp['total_size'],
    stratified= exp['stratified'],
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
    init_min = exp['init_min'],
    initialization= exp['initialization'],
    final_init = exp['final_init']
)

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
