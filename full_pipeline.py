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
from modules.test import final_test
from uqmethods.al import DatasetAL

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

def main(exp_config, model_config):
    
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
        adapt_weights = model_config['adapt_weights'],
        learn_residual=model_config["learn_residual"], 
        hidden_sizes=model_config["hidden_sizes"], 
        activation=model_config["activation"],
        solver=model_config["solver"],
        initialization=model_config['initialization'],

        # uq args
        n_models = model_config['n_models'],
        uq_method=model_config['uq_method'],
        n_modes=model_config['n_modes'],
        hidden_sizes_uq=model_config['hidden_sizes_uq'],
    )

    # Params initializing or restoring
    model, checkpointer = initialize_or_restore_params(model, True, model_config["model_name"], base_dir= "experiments/" + exp_config['exp_name'] + "/weights", rank=rank, seed=exp_config['seed']) # check or do it deeper

    # Ingest data
    if exp_config['al']:

        dataset_al = DatasetAL(exp_config['M'], exp_config['N'], exp_config['K'], exp_config['T'], exp_config['dynamic_query'], exp_config['test_size'], exp_config['convergence'], exp_config['exp_name'], model_config['model_name'], exp_config['seed'])
        dataset_train = dataset_al.initialize(rank)
        dataset_test, dataset_val = dataset_al.get_other_set()

    else:

        dataset_train, dataset_test, dataset_val  = data_ingestion(
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

        model, mse_train, mse_val, perc_error_val = train_model(
            exp_name=exp_config['exp_name'], 
            model_name=model_config["model_name"],
            dataset_al= dataset_al,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            model_real=model,
            stop_perc = exp_config['stop_perc'],
            loss_type= model_config["uq_method"],
            beta_loss = exp_config["loss_beta"],    
            learn_rate_min=exp_config["learn_rate_min"], 
            learn_rate_max=exp_config["learn_rate_max"], 
            schedule=exp_config["schedule"], 
            epochs=exp_config["epochs"], 
            batch_size=exp_config["batch_size"],
            checkpointer=checkpointer,
            debug=False
            )
        
        model, checkpointer = initialize_or_restore_params(model, False, model_config["model_name"], base_dir= "experiments/" + exp_config['exp_name'] + "/weights", rank=rank, seed=exp_config['seed']) # check or do it deeper

        if rank == 0:
            final_test(exp_config['exp_name'], model, model_config['model_name'], dataset_test, mse_train, mse_val, perc_error_val)


    model, checkpointer = initialize_or_restore_params(model, False,  model_config["model_name"], base_dir= "experiments/" + exp_config['exp_name'] + "/weights", rank=rank, seed=exp_config['seed']) # check or do it deeper


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
            stochastic=exp_config['stochastic'],
            seed=rngs
        )


if __name__ == "__main__":

    import sys
    from config_experiment import basic_1000_train, dataeff_100_train, dataeff_200_train, dataeff_500_train, earlystop
    from config_model import peds_fourier, peds_gauss, mlpmod, peds_f_ens, peds_g_ens, peds_f_ens_uq1, peds_g_ens_uq1

    # Dictionaries to map names to objects
    experiments = {
        "1000_data": basic_1000_train,
        "100_data": dataeff_100_train,
        "200_data": dataeff_200_train,
        "500_data": dataeff_500_train,
        "earlystop": earlystop,
    }

    models = {
        "peds_fourier": peds_fourier,
        "peds_gauss": peds_gauss,
        "mlpmod": mlpmod,
        "peds_fourier_ens": peds_f_ens,
        "peds_gauss_ens": peds_g_ens,
        "peds_fourier_uq1": peds_f_ens_uq1,
        "peds_gauss_uq1": peds_g_ens_uq1
    }

    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <experiment_name> <model_name>")
        print(f"Available experiments: {list(experiments.keys())}")
        print(f"Available models: {list(models.keys())}")
        sys.exit(1)

    exp_name = sys.argv[1]
    model_name = sys.argv[2]

    # Look up the actual objects
    exp = experiments.get(exp_name)
    model = models.get(model_name)

    if exp is None or model is None:
        print("Invalid experiment or model name.")
        print(f"Available experiments: {list(experiments.keys())}")
        print(f"Available models: {list(models.keys())}")
        sys.exit(1)

    # Now run the main function
    main(exp, model)



