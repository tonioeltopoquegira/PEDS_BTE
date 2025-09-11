import numpy as np
from models.model_utils import predict
import os

def final_test(exp_seed, exp_name, model, model_name, dataset, mse_train, mse_val, perc_error_val):
    pores, kappa = dataset
    pores = pores.reshape((pores.shape[0], 25))
    
    kappa_pred, kappa_var = predict(model, pores, training=False)

    # Compute overall percentage error
    error = np.abs(kappa_pred - kappa) / np.abs(kappa_pred)
    overall_error = error.mean().item() * 100.0

    # Threshold-based splits
    split_ranges = {
        "<15": kappa < 15,
        "15-40": (kappa >= 15) & (kappa < 40),
        ">40": kappa >= 40
    }

    split_errors = {}
    for label, mask in split_ranges.items():
        if np.any(mask):
            split_error = np.mean(np.abs(kappa_pred[mask] - kappa[mask]) / np.abs(kappa_pred[mask])) * 100.0
            split_errors[label] = split_error
        else:
            split_errors[label] = np.nan  # Mark as NaN

    # Quartile-based splits
    q1, q2, q3 = np.quantile(kappa, [0.25, 0.5, 0.75])
    quartile_ranges = {
        "Q1": kappa <= q1,
        "Q2": (kappa > q1) & (kappa <= q2),
        "Q3": (kappa > q2) & (kappa <= q3),
        "Q4": kappa > q3,
    }

    quartile_errors = {}
    for label, mask in quartile_ranges.items():
        if np.any(mask):
            q_error = np.mean(np.abs(kappa_pred[mask] - kappa[mask]) / np.abs(kappa_pred[mask])) * 100.0
            quartile_errors[label] = q_error
        else:
            quartile_errors[label] = np.nan

    # Prepare file
    result_path = f"experiments/{exp_name}/results/errors.txt"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # Write header if file is empty
    if not os.path.exists(result_path) or os.stat(result_path).st_size == 0:
        with open(result_path, "w") as f:
            f.write(
                "model_name,mse_train,mse_val,val_perc_error,overall_test_perc_error,"
                "test_perc_error_<15,test_perc_error_15-40,test_perc_error_>40,"
                "test_perc_error_Q1,test_perc_error_Q2,test_perc_error_Q3,test_perc_error_Q4\n"
            )

    # Append data row
    with open(result_path, "a") as f:
        f.write(
            f"{model_name},{mse_train:.6f},{mse_val:.6f},{perc_error_val:.4f},{overall_error:.4f},"
            f"{split_errors['<15']:.4f},{split_errors['15-40']:.4f},{split_errors['>40']:.4f},"
            f"{quartile_errors['Q1']:.4f},{quartile_errors['Q2']:.4f},"
            f"{quartile_errors['Q3']:.4f},{quartile_errors['Q4']:.4f}\n"
        )

    print(f'Test error: {overall_error}')
