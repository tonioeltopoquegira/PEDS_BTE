import os
import jax.numpy as jnp
from mpi4py import MPI
from flax import nnx

from modules.data_ingestion import data_ingestion
from models.model_utils import select_model
from modules.params_utils import initialize_or_restore_params
from modules.training import train_model
from utils import create_folders
from optimization.run_optimization import optimize
from modules.validation import final_validation
from uqmethods.al import DatasetAL

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from config_experiment import basic_1000_train as exp_config
from config_model import mlp_ens as model_config # change to m2 for ensemble!


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize random key
rngs = nnx.Rngs(exp_config["seed"])

create_folders(exp_config['exp_name'], model_config['model_name']) # experiment name

# Select, create and initialize models
model = select_model(
    seed=exp_config["seed"],    
    model_type=model_config["model"], 
    resolution=model_config["resolution"], 
    learn_residual=model_config["learn_residual"], 
    hidden_sizes=model_config["hidden_sizes"], 
    activation=model_config["activation"],
    solver=model_config["solver"],
    initialization=model_config['initialization'],
    n_models = model_config['n_models']
)

# Params initializing or restoring
model, checkpointer = initialize_or_restore_params(model, model_config["model_name"], base_dir= "experiments/" + exp_config['exp_name'] + "/weights", rank=rank, seed=exp_config['seed']) # check or do it deeper

# Ingest data
if exp_config['al']:

    dataset_al = DatasetAL(exp_config['M'], exp_config['N'], exp_config['K'], exp_config['T'], exp_config['exp_name'], model_config['model_name'], exp_config['seed'])
    dataset_train = dataset_al.initialize(rank)
    dataset_test = dataset_al.get_test_set()

else:

    dataset_train, dataset_test = data_ingestion(
        rank=rank,
        exp_name=exp_config['exp_name'],
        train_size=exp_config["train_size"], 
        test_size=exp_config['test_size'],
        key=rngs
    )

    dataset_al = None

if exp_config['training']:

    if rank == 0:
        print(f"Training on {size} devices...")

    model, mse_train, mse_test, perc_error_test = train_model(
        exp_name=exp_config['exp_name'], 
        model_name=model_config["model_name"],
        dataset_al= dataset_al,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        model_real=model,
        learn_rate_min=exp_config["learn_rate_min"], 
        learn_rate_max=exp_config["learn_rate_max"], 
        schedule=exp_config["schedule"], 
        epochs=exp_config["epochs"], 
        batch_size=exp_config["batch_size"],
        checkpointer=checkpointer,
        debug=True
        )
      
    model, checkpointer = initialize_or_restore_params(model, model_config["model_name"], base_dir= "experiments/" + exp_config['exp_name'] + "/weights", rank=rank, seed=exp_config['seed']) # check or do it deeper

    if rank == 0:
        final_validation(exp_config['exp_name'], model, model_config['model_name'], dataset_test, mse_train, mse_test, perc_error_test)


model, checkpointer = initialize_or_restore_params(model, model_config["model_name"], base_dir= "experiments/" + exp_config['exp_name'] + "/weights", rank=rank, seed=exp_config['seed']) # check or do it deeper


if rank == 0 and exp_config['optimization']:
    
    design_data = jnp.load("data/highfidelity/design_data.npz", allow_pickle=True)
    pores_truth = jnp.asarray(design_data['pores'], dtype=jnp.float32)
    kappa_target = jnp.asarray(design_data['kappas'], dtype=jnp.float32)

    optimize(
        exp_name=exp_config['exp_name'],
        model_name=model_config["model_name"],
        model=model,
        opt=exp_config['opt'],
        kappas= kappa_target,
        seed=rngs
    )