#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# ------------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------------
data = np.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
X = data["pores"].astype(np.float32)    # shape (N,25)
y = data["kappas"].astype(np.float32)   # shape (N,)

N = len(X)
print(f"Loaded dataset: {X.shape}, {y.shape}")

# ------------------------------------------------------------------
# 2) Experiment configuration
# ------------------------------------------------------------------
n_train = 1000
n_val   = 1000

# Function to get indices for a given split i (train) and j (test)
def get_split_indices(i, j, random_state=0):
    rng = np.random.default_rng(random_state)

    # Partition into groups by κ value
    mask0 = y <= 20
    mask1 = (y > 20) & (y <= 45)
    mask2 = y > 45

    idx0 = np.where(mask0)[0]
    idx1 = np.where(mask1)[0]
    idx2 = np.where(mask2)[0]

    groups = [idx0, idx1, idx2]

    train_idx = rng.choice(groups[i], size=n_train, replace=False)
    test_idx  = rng.choice(groups[j], size=n_val, replace=False)

    return train_idx, test_idx

# ------------------------------------------------------------------
# 3) GP training/eval function
# ------------------------------------------------------------------
def train_eval_gp(train_idx, test_idx):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-6)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=43
    ).fit(X_train, y_train)

    y_pred = gp.predict(X_test)
    frac_error = np.abs((y_pred - y_test) / y_test)
    val_perc_error = 100 * np.mean(frac_error)  # percentage mean fractional error
    return val_perc_error

# ------------------------------------------------------------------
# 4) Build 3x3 GP matrix
# ------------------------------------------------------------------
mat_gp = np.zeros((3,3))

for i in range(3):  # training set group
    for j in range(3):  # test set group
        if i == j:
            mat_gp[i, j] = np.nan
            continue
        train_idx, test_idx = get_split_indices(i, j)
        val_error = train_eval_gp(train_idx, test_idx)
        mat_gp[i, j] = val_error
        print(f"Split ({i},{j}): val_perc_error={val_error:.2f}%")

# ------------------------------------------------------------------
# 5) Save results for later plotting
# ------------------------------------------------------------------
outdir = "experiments/gp_results"
os.makedirs(outdir, exist_ok=True)

# Save as CSV
pd.DataFrame(mat_gp).to_csv(os.path.join(outdir, "gp_val_perc_error_matrix.csv"), index=False, header=False)

# Save as NumPy file too
np.save(os.path.join(outdir, "gp_val_perc_error_matrix.npy"), mat_gp)

print(f"\nSaved GP matrix to {outdir}/gp_val_perc_error_matrix.csv and .npy")
