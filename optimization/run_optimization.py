import numpy as np
import pandas as pd
import os 

from optimization.ga import genetic_algorithm
from optimization.gradient import gradient_opt

#from ga import genetic_algorithm
#from gradient import gradient_opt



def optimize(exp_name, model_name, model, opt, kappas, stochastic, seed):

    print(f"Start Optimization with {opt} for {kappas}... \n")

    if stochastic:
        results_file = f"experiments/{exp_name}/optimization/{opt}_{model_name}_stochastic.csv"
    else:
        results_file = f"experiments/{exp_name}/optimization/{opt}_{model_name}.csv"

    # Load existing results if the file exists, otherwise create a new DataFrame
    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
    else:
        results = pd.DataFrame(columns=["kappa_target", "kappa_optimized", "error_optimization", "geometry"])

    optimizer = choose_optimizer(opt)

    for k in kappas:

        design, kappa_optimized, var_optimized = optimizer(model, k, stochastic, seed)

        if var_optimized is None:
            var_optimized= 9999.0

        print(f"Optimized for {k}: found {kappa_optimized} w/ {var_optimized} for {design}")

        # Append the results to the DataFrame
        results = results._append({"kappa_target": k, "kappa_optimized": kappa_optimized.item(), "var_optimized": var_optimized.item(), "error_optimization": np.abs((k - kappa_optimized)/(k)).item(), "geometries": design.tolist()}, ignore_index=True)
    
    # Save the DataFrame to the CSV file after every iteration
    results.to_csv(results_file, index=False)

    print("Optimizations completed.")


def choose_optimizer(opt):

    if opt == "ga":
        return lambda model, k, stochastic, seed: genetic_algorithm(model, k, stochastic, seed,  n=25, pop_size=500, generations=60, cxpb=0.5, mutpb=0.2, tournsize=3, indpb=0.05)
    
    if opt == "grad":
        return lambda model, k, stochastic, seed: gradient_opt(model, k, stochastic, seed, steps=100, lr=0.1)
    
    else:
        print("Unrecognized optimization method")
        pass
        

if __name__ == "__main__":

    import os


    from models.peds import PEDS
    from modules.params_utils import initialize_or_restore_params
    from models.model_utils import select_model
    from flax import nnx


    kappas = [45.0]

    rngs = nnx.Rngs(42)

    
    from config_model import peds_fourier_ens as model_config 
    seed = nnx.Rngs(42)

   # Select, create and initialize models
    model = select_model(
        seed=42,    
        model_type=model_config["model"], 
        resolution=model_config["resolution"], 
        adapt_weights = model_config['adapt_weights'],
        learn_residual=model_config["learn_residual"], 
        hidden_sizes=model_config["hidden_sizes"], 
        activation=model_config["activation"],
        solver=model_config["solver"],
        initialization=model_config['initialization'],
        n_models = model_config['n_models']
    )

    # Params initializing or restoring
    model, checkpointer = initialize_or_restore_params(model, model_config["model_name"], base_dir= "experiments/train_1000/weights", rank=0, seed=42) # check or do it deeper

   

    optimize(exp_name="train_1000", model_name=model_config["model_name"], model=model, opt="grad", kappas=kappas, stochastic=True, seed=seed)



