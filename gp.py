#!/usr/bin/env python3
import numpy as np
from numpy import random
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")

# -----------------------------------------------------------------------------
# 1) LOAD YOUR DATA AND SPLIT TRAIN / TEST
# -----------------------------------------------------------------------------
data = np.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
X = data["pores"].astype(np.float32)    # shape (N,25)
y = data["kappas"].astype(np.float32)   # shape (N,)

# fix random seed for reproducibility
rng = np.random.default_rng(42)
perm = rng.permutation(len(X))

# hold out first 1000 as test, the rest as training
test_idx = perm[:1000]
train_idx = perm[1000:1300]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

'''plt.figure(figsize=(8,5))
sns.kdeplot(y_train, label='y_train', fill=True, alpha=0.5)
sns.kdeplot(y_test, label='y_test', fill=True, alpha=0.5)
plt.xlabel('y values')
plt.ylabel('Density')
plt.title('Density of y_train and y_test')
plt.legend()
plt.show()'''

print(f"Training on {len(X_train)} points, testing on {len(X_test)} points")

# -----------------------------------------------------------------------------
# 2) FIT A GAUSSIAN PROCESS REGRESSOR
# -----------------------------------------------------------------------------
kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e1)) # RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=43,
).fit(X_train, y_train)
print("GP fitted.  Learned kernel:", gp.kernel_)

# -----------------------------------------------------------------------------
# 3) EVALUATE SURROGATE ACCURACY ON HOLD‑OUT TEST SET
# -----------------------------------------------------------------------------
y_pred, y_std = gp.predict(X_test, return_std=True)
mse = mean_squared_error(y_test, y_pred)
relative_mse = mse / np.var(y_test)  # relative to variance of y_test
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test MSE: {mse:.4f}, Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")
print(f"Relative MSE: {relative_mse:.4f}")

frac_error = np.abs((y_pred - y_test) / y_test)
print(f"Mean fractional error: {100 * np.mean(frac_error):.2f}%")
print(f"Max  fractional error: {100 * np.max(frac_error):.2f}%")

q1, q2, q3 = np.percentile(y_test, [25, 50, 75])

# Masks for each quartile
mask_q1 = y_test <= q1
mask_q2 = (y_test > q1) & (y_test <= q2)
mask_q3 = (y_test > q2) & (y_test <= q3)
mask_q4 = y_test > q3

# Compute mean fractional error per quartile
frac_q1 = np.mean(frac_error[mask_q1])
frac_q2 = np.mean(frac_error[mask_q2])
frac_q3 = np.mean(frac_error[mask_q3])
frac_q4 = np.mean(frac_error[mask_q4])

print("Mean fractional error per quartile:")
print(f"Q1 (<= {q1:.2f}): {100*frac_q1:.2f}%")
print(f"Q2 ({q1:.2f}-{q2:.2f}): {100*frac_q2:.2f}%")
print(f"Q3 ({q2:.2f}-{q3:.2f}): {100*frac_q3:.2f}%")
print(f"Q4 (> {q3:.2f}): {100*frac_q4:.2f}%")

mask_low  = y_test < 15
mask_mid  = (y_test >= 15) & (y_test <= 40)
mask_high = y_test > 40

frac_low  = frac_error[mask_low]
frac_mid  = frac_error[mask_mid]
frac_high = frac_error[mask_high]

print("\nMean ± std fractional error by κ range:")
print(f"κ < 15      : {100*np.mean(frac_low):.2f}%%")
print(f"15 ≤ κ ≤ 40 : {100*np.mean(frac_mid):.2f}% ")
print(f"κ > 40      : {100*np.mean(frac_high):.2f}% ")

# -----------------------------------------------------------------------------
# 4) DEFINE HELPER TO “INVERT” THE GP TO A DESIRED TARGET
# -----------------------------------------------------------------------------
def find_pattern_for_target(
    gp: GaussianProcessRegressor,
    target: float,
    n_restarts: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Find a binary 5×5 pattern (flattened to length‑25) whose
    GP‐predicted mean is as close as possible to `target`.

    Returns
    -------
    best_x_bin : np.ndarray, shape (25,), dtype=int
        The binary pattern (0/1) that best matches the target.
    best_loss : float
        The squared‐error (μ_GP(x) - target)^2 at that point.
    """
    best_res = None
    for _ in range(n_restarts):
        x0 = rng.random(25)  # continuous start in [0,1]^25
        res = minimize(
            fun=lambda x: (gp.predict(x.reshape(1, -1))[0] - target) ** 2,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * 25,
            options={"maxiter": 200},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    # round continuous solution to binary
    best_x_bin = (best_res.x > 0.5).astype(int)
    best_loss = float(best_res.fun)
    return best_x_bin, best_loss

# -----------------------------------------------------------------------------
# 5) DEMO INVERSION FOR A FEW TARGETS
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for T in [12.01]: # 15.00, 20.00, 30.00, 44.98, 59.99
        #pattern25, loss = find_pattern_for_target(gp, target=T, n_restarts=20)
        #print(f"\nTarget kappa = {T}")
        #print(" Best 1D pattern:", pattern25.tolist())
        #print(f" Surrogate‐predicted squared error: {loss:.6f}")
        continue


# 12.01 , [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0.000000
# 15.00 , [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], 0.0000
# 20.00 ,  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], 0.000
# 30.00, [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0], 0.00
# 44.98 , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.00
# 59.99, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 12.01 ,[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 15.00 [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], 0.
# 20.00 [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1] 0.0
# 30.00, [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 0.00
# 44.98 , [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.00
# 59.99, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.00


# Genetic Algorithm

