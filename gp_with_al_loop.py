#!/usr/bin/env python3
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")

# -------------------------
# 1) LOAD DATA
# -------------------------
data = np.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
X_all = data["pores"].astype(np.float32)    # (N,25)
y_all = data["kappas"].astype(np.float32)   # (N,)

rng = np.random.default_rng(44)
perm = rng.permutation(len(X_all))
X_all, y_all = X_all[perm], y_all[perm]

# Define test set
test_size = 1000
X_test, y_test = X_all[:test_size], y_all[:test_size]

# Candidate pool (exclude test set)
X_pool, y_pool = X_all[test_size:], y_all[test_size:]

# -------------------------
# 2) ACTIVE LEARNING LOOP
# -------------------------
total_size = 2000   # final training dataset size
batch_fraction = 1/2  # fraction of final size per iteration
sample_factor = 8     # how many candidates to sample per batch

# GP kernel
kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-6)

# Start with empty training dataset
X_train = np.empty((0, X_all.shape[1]), dtype=np.float32)
y_train = np.empty((0,), dtype=np.float32)

while len(X_train) < total_size:
    # Determine batch size and sample size
    batch_size = int(total_size * batch_fraction)
    batch_size = min(batch_size, total_size - len(X_train))  # prevent overshoot
    sample_size = batch_size * sample_factor

    # Randomly sample candidates from the pool
    sample_idx = rng.choice(len(X_pool), size=min(sample_size, len(X_pool)), replace=False)
    X_candidates = X_pool[sample_idx]
    y_candidates = y_pool[sample_idx]

    if len(X_train) > 0:
        # Fit GP on current training set
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
        gp.fit(X_train, y_train)

        # Predict uncertainty on candidates
        _, std = gp.predict(X_candidates, return_std=True)
        # Select the batch_size most uncertain points
        uncertain_idx = np.argsort(-std)[:batch_size]
    else:
        # First iteration: just take first batch_size points
        uncertain_idx = np.arange(min(batch_size, len(X_candidates)))

    # Add selected points to training set
    X_train = np.vstack([X_train, X_candidates[uncertain_idx]])
    y_train = np.concatenate([y_train, y_candidates[uncertain_idx]])

    # Remove selected points from pool
    X_pool = np.delete(X_pool, sample_idx[uncertain_idx], axis=0)
    y_pool = np.delete(y_pool, sample_idx[uncertain_idx], axis=0)

    print(f"Training set size: {len(X_train)} / {total_size}")

# -------------------------
# 3) TRAIN FINAL GP
# -------------------------
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
gp.fit(X_train, y_train)
print("GP fitted. Learned kernel:", gp.kernel_)

# -------------------------
# 4) EVALUATE
# -------------------------
y_pred, y_std = gp.predict(X_test, return_std=True)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
relative_mse = mse / np.var(y_test)
frac_error = np.abs((y_pred - y_test) / y_test)
mean_frac_error = 100 * np.mean(frac_error)
max_frac_error = 100 * np.max(frac_error)

print(f"Mean fractional error: {mean_frac_error:.2f}%")
print(f"Max  fractional error: {max_frac_error:.2f}%")
