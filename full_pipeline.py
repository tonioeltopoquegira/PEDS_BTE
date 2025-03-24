import os
from mpi4py import MPI
from flax import nnx
from config import config as exp  

from modules.data_ingestion import data_ingestion
from modules.choose_model import select_model
from modules.params_utils import initialize_or_restore_params
from modules.training import train_model
from modules.training_utils import create_folders
from modules.run_optimization import optimize
from modules.training_utils import final_validation

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize random key
rngs = nnx.Rngs(exp["seed"])

create_folders(exp['model_name'])

model = select_model(
    rngs=rngs, 
    model_type=exp["model"], 
    resolution=exp["resolution"], 
    learn_residual=exp["learn_residual"], 
    hidden_sizes=exp["hidden_sizes"], 
    activation=exp["activation"],
    solver=exp["solver"],
    init_min=exp['init_min'],
    initialization=exp['initialization'],
    reg=exp['reg'],
    final_init=exp['final_init']
)

# Params initializing or restoring
model, checkpointer = initialize_or_restore_params(model, exp["model_name"], rank=rank)

# Ingest data
dataset_train, dataset_test, kappas_valid = data_ingestion(
    rank=rank,
    model_name=exp['model_name'],
    filename=exp["filename_data"], 
    train_size=exp["train_size"], 
    total_size=exp['total_size'],
    stratified=exp['stratified'],
    key=rngs
)

if exp['training']:
    if rank == 0:
        print(f"Training on {size} devices...")
    train_model(
        exp=exp, 
        model_name=exp["model_name"],
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        model=model,
        reg=exp['reg'],
        learn_rate_min=exp["learn_rate_min"], 
        learn_rate_max=exp["learn_rate_max"], 
        schedule=exp["schedule"], 
        epochs=exp["epochs"], 
        batch_size=exp["batch_size"],
        checkpointer=checkpointer,
        print_every=exp["print_every"]
    )

if exp['valid'] and rank == 0:
    if exp['new_robust_check']:
        dataset_train, dataset_test, kappas_valid = data_ingestion(
            0, exp['model_name'], exp['filename_data'], exp['total_size'], exp['train_size'], "small->small", rngs
        )
    print("Validation...")
    final_validation(exp, model, exp['model_name'], dataset_test)

if rank == 0 and exp['optimization']:
    optimize(
        model_name=exp["model_name"],
        model=model,
        opt=exp['opt'],
        kappas=exp.get('kappas', kappas_valid),
        seed=rngs
    )

# ToDo:

# 1 Create a configuration file that is separate...
# 2 Adjust file system with experiment names
# 3 Create comparison between models automated
# 4 Adjust ingestion with dropout
# 5 Add uncertainty to each model (ideally put it inside each model like you did with generator+solver)
# 6 Same but with ensemble (store N models)


# After break

# Active Learning framework
# Stochastic optimization (easy)
# Compare results with and without UQ