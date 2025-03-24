import numpy as np
import pandas as pd
import os 


from optimization.ga import genetic_algorithm
from optimization.gradient import gradient_opt




def optimize(model_name, model,opt, kappas, seed):

    print(f"Start Optimization with {opt} for {kappas}... \n")

    if "PEDS" in model_name:
        out = "peds"
    else:
        out = "mlp"

    os.makedirs(f"data/optimization/{model_name}", exist_ok=True)
    results_file = f"data/optimization/{model_name}/{opt}_geometries_errors.csv"

    # Load existing results if the file exists, otherwise create a new DataFrame
    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
    else:
        results = pd.DataFrame(columns=["kappa_target", "kappa_optimized", "error_optimization", "geometry"])

    optimizer = choose_optimizer(opt)

    for k in kappas:

        design, kappa_optimized = optimizer(model, k, out, seed)

        print(f"Optimized for {k}: found {kappa_optimized} for {design}")

        # Append the results to the DataFrame
        results = results._append({"kappa_target": k, "kappa_optimized": kappa_optimized.item(), "error_optimization": np.abs((k - kappa_optimized)/(k)).item(), "geometries": design.tolist()}, ignore_index=True)

    
    # Save the DataFrame to the CSV file after every iteration
    results.to_csv(results_file, index=False)

    print("Optimizations completed.")


def choose_optimizer(opt):

    if opt == "ga":
        return lambda model, k, out, seed: genetic_algorithm(model, k, out, seed,  n=25, pop_size=200, generations=40, cxpb=0.5, mutpb=0.2, tournsize=3, indpb=0.05)
    
    if opt == "grad-adam":
        return lambda model, k, out, seed: gradient_opt(model, k, out, seed,  neigh=True, batch_size=200, steps=50, lr=0.1)
    
    else:
        print("Unrecognized optimization method")
        pass
        

if __name__ == "__main__":

    import os
    os.chdir("/Users/antoniovaragnolo/Desktop/PEDSBoltzmann/Codes/")
    print("Current working directory:", os.getcwd())


    from models.peds import PEDS
    from modules.params_utils import initialize_or_restore_params
    from flax import nnx


    kappas = [0.0, 160.0]

    model_name = "PEDS_gauss"
    # Initialize the model
    model = PEDS(resolution = 20, learn_residual= False, hidden_sizes= [32, 64, 128], activation="relu", solver="gauss", final_init=False, initialization="he", init_min=1e-11) # parameters: 60k
    model, checkpointer = initialize_or_restore_params(model, model_name, rank=0)



    seed = nnx.Rngs(42)

    optimize(model_name, model, "grad-adam", kappas, seed)



