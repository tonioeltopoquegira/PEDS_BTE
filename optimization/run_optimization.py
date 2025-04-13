import numpy as np
import pandas as pd
import os 

from optimization.ga import genetic_algorithm
from optimization.gradient import gradient_opt

#from ga import genetic_algorithm
#from gradient import gradient_opt



def optimize(exp_name, model_name, model,opt, kappas, seed):

    print(f"Start Optimization with {opt} for {kappas}... \n")

   
    # results_file = f"experiments/{exp_name}/optimization/{opt}_{model_name}.csv"
    results_file = f"{exp_name}_{opt}_{model_name}.csv"

    # Load existing results if the file exists, otherwise create a new DataFrame
    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
    else:
        results = pd.DataFrame(columns=["kappa_target", "kappa_optimized", "error_optimization", "geometry"])

    optimizer = choose_optimizer(opt)

    for k in kappas:

        design, kappa_optimized = optimizer(model, k, seed)

        print(f"Optimized for {k}: found {kappa_optimized} for {design}")

        # Append the results to the DataFrame
        results = results._append({"kappa_target": k, "kappa_optimized": kappa_optimized.item(), "error_optimization": np.abs((k - kappa_optimized)/(k)).item(), "geometries": design.tolist()}, ignore_index=True)
    
    # Save the DataFrame to the CSV file after every iteration
    results.to_csv(results_file, index=False)

    print("Optimizations completed.")


def choose_optimizer(opt):

    if opt == "ga":
        return lambda model, k,seed: genetic_algorithm(model, k, seed,  n=25, pop_size=200, generations=40, cxpb=0.5, mutpb=0.2, tournsize=3, indpb=0.05)
    
    if opt == "grad":
        return lambda model, k, seed: gradient_opt(model, k,seed,  neigh=False, batch_size=200, steps=400, lr=0.1)
    
    if opt == "grad_var":
        return lambda model, k, seed: gradient_opt(model, k,seed,  neigh=False, min_var=True, batch_size=200, steps=200, lr=0.1)

    if opt == 'smoothed':
        return lambda model, k, seed: gradient_opt(model, k,seed,  neigh=False, min_var=True, use_smoothed=True, use_penalty=False, batch_size=200, steps=200, lr=0.1)
    
    else:
        print("Unrecognized optimization method")
        pass
        

if __name__ == "__main__":

    import os
    # os.chdir("/Users/antoniovaragnolo/Desktop/PEDSBoltzmann/Codes/")
    print("Current working directory:", os.getcwd())


    from models.peds import PEDS
    from modules.params_utils import initialize_or_restore_params
    from models.model_utils import select_model
    from flax import nnx


    kappas = [12.0]

    rngs = nnx.Rngs(42)

    
    from config_model import m1 as model_config 
    # from config_model import m2 as model_config # Change this to m2 for ENSEMBLE model (UQ)

    seed = nnx.Rngs(42)

    model = select_model(
        seed=42,
        rngs=rngs, 
        model_type=model_config["model"], 
        resolution=model_config["resolution"], 
        learn_residual=model_config["learn_residual"], 
        hidden_sizes=model_config["hidden_sizes"], 
        activation=model_config["activation"],
        solver=model_config["solver"],
        initialization=model_config['initialization'],
        n_models = model_config['n_models']
    )

    model, checkpointer = initialize_or_restore_params(model,model_config["model_name"], base_dir= "experiments/opt_coding/weights", rank=0, seed=42) # check or do it deeper

    # optimize("test_exp", model_config["model_name"], model, "grad_var", kappas, seed)
    optimize("test_exp", model_config["model_name"], model, "smoothed", kappas, seed)
    # optimize("test_exp", model_config["model_name"], model, "grad-adam", kappas, seed)
