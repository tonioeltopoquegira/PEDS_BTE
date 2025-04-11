import pandas as pd
import numpy as np
import argparse
from OpenBTE_highfid import highfidelity_solver

def run_optimization(exp_name, model_name, optimizer):
    # Load results
    results = pd.read_csv(f"experiments/{exp_name}/optimization/{optimizer}_{model_name}.csv")
    result_new = pd.DataFrame(columns=["kappa_target", "kappa_optimized", "error_optimization", "geometry", "kappa_BTE", "error_model"])

    for _, res in results.iterrows():
        k, k_opt, error_opt, _, geom = res

        # Convert to list of integers
        design = geom.strip("\"").strip("[]")  # Remove quotes and brackets
        design = np.array([int(x) for x in design.split(", ")]) 

        # Reshape into 5x5
        if design.size == 25:
            pores = design.reshape((5, 5))
        else:
            raise ValueError(f"Invalid design shape: expected 25 elements, got {design.size}")

        # Run high-fidelity solver
        kappa, _, _ = highfidelity_solver(pores, step_size=2, save_show_res=False)

        # Compute error
        error_model = np.abs((kappa - k_opt) / k_opt).item()
        error_tot = np.abs((kappa - k) / k).item()

        # Store results
        result_new = result_new._append({
            "kappa_target": k,
            "kappa_optimized": k_opt,
            "geometry": geom,
            "kappa_BTE": kappa,
            "error_optimization": error_opt,
            "error_model": error_model,
            "error_total": error_tot
        }, ignore_index=True)

    # Save updated results
    result_new.to_csv(f"experiments/{exp_name}/optimizations/{optimizer}_{model_name}_withBTE.csv", index=False)
    print(f"Optimization results saved to experiments/{exp_name}/optimizations/{optimizer}_{model_name}_withBTE.csv")

if __name__ == "__main__":

    exp_name = "train_1000"

    model_names = ["peds_mixed_smaller", "peds_mixed_smaller1", "peds_mixed_smaller2", "peds_mixed_smaller3", "peds_mixed_smaller4",
                    "mlp", "mlp1", "mlp2", "mlp3", "mlp4"]

    optimizer = "ga"
    for m in model_names:
        run_optimization(exp_name, m, optimizer)

