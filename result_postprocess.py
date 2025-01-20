import numpy as np

# Ingest data <- Here we will do active learning
full_data = np.load("data/highfidelity/high_fidelity_10012_20steps.npz", allow_pickle=True)

pores = np.asarray(full_data['pores'], dtype=np.float32)
kappas = np.asarray(full_data['kappas'], dtype=np.float32)
base_conductivities = np.asarray(full_data['conductivity'], dtype=np.float32)

# Create dataset
dataset_train = [pores[:8000], base_conductivities[:8000], kappas[:8000]]
dataset_valid = [pores[8000:], base_conductivities[8000:], kappas[8000:]]

from models.mlp import mlp
from flax import nnx
import os
import orbax.checkpoint as ocp

from modules.params_utils import initialize_or_restore_params

model_name = "MLP_baseline"

rngs = nnx.Rngs(42)
model = mlp(layer_sizes=[25, 32, 64, 128, 128, 256, 1], activation="relu", rngs=rngs)

model, checkpointer, ckpt_dir = initialize_or_restore_params(model, model_name, rank=0)

pores_valid, cond_valid, kappa_valid = dataset_valid

pores_valid = pores_valid.reshape((pores_valid.shape[0], 25))

kappa_pred = model(pores_valid)

kappa_pred = kappa_pred.squeeze(-1)

# I want to plot the distribution of error = np.abs(kappa_pred - kappa_valid) / np.abs(kappa_valid). Plot this value against the respective kappa_valid

import matplotlib.pyplot as plt

# Compute the error
error = np.abs(kappa_pred - kappa_valid) / np.abs(kappa_valid)

print(kappa_valid.shape)
print(error.shape)

# Plot the error against kappa_valid
plt.figure(figsize=(10, 6))
plt.scatter(kappa_valid, error, alpha=0.5, s=10, c='blue', label="Error Distribution")
plt.xlabel(r"$\kappa_{\text{valid}}$", fontsize=14)
plt.ylabel(r"Relative Error: $\frac{| \kappa_{\text{pred}} - \kappa_{\text{valid}} |}{| \kappa_{\text{valid}} |}$", fontsize=14)
plt.title("Relative Error vs. True Conductivity ($\kappa_{\text{valid}}$)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Bin data
num_bins = 20
bins = np.linspace(kappa_valid.min(), kappa_valid.max(), num_bins)
bin_indices = np.digitize(kappa_valid, bins)

# Compute mean error for each bin
mean_errors = [error[bin_indices == i].mean() for i in range(1, len(bins))]
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_errors, marker='o', label="Mean Error (Binned)", color='red')
plt.xlabel(r"$\kappa_{\text{valid}}$", fontsize=14)
plt.ylabel("Mean Relative Error", fontsize=14)
plt.title("Mean Relative Error vs. True Conductivity ($\kappa_{\text{valid}}$)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# Now I have pores_valid (binary vector of size 25) and kappa_valid (continuous value) and I want to do a sort of PCA to see if kappa_valid can be explained easily by some of the components 

from sklearn.decomposition import PCA

# Perform PCA
n_components = 10  # Number of components to keep
pca = PCA(n_components=n_components)
pores_pca = pca.fit_transform(pores_valid)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, align='center', label="Explained Variance Ratio")
plt.step(range(1, n_components + 1), np.cumsum(explained_variance), where='mid', label="Cumulative Variance")
plt.xlabel("Principal Component", fontsize=14)
plt.ylabel("Explained Variance Ratio", fontsize=14)
plt.title("PCA Explained Variance", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_ind, spearmanr, pointbiserialr


# Example data
pores_valid = np.random.randint(0, 2, (100, 25))  # Binary features
kappa_valid = np.random.rand(100)  # Continuous target

# Statistical Analysis
t_tests = []
spearman_corr = []
point_biserial_corr = []
for i in range(pores_valid.shape[1]):
    group0 = kappa_valid[pores_valid[:, i] == 0]
    group1 = kappa_valid[pores_valid[:, i] == 1]
    
    # t-test
    t_stat, p_value = ttest_ind(group0, group1, equal_var=False)
    t_tests.append((t_stat, p_value))
    
    # Spearman correlation
    corr, _ = spearmanr(pores_valid[:, i], kappa_valid)
    spearman_corr.append(corr)
    
    # Point-biserial correlation
    pb_corr, _ = pointbiserialr(pores_valid[:, i], kappa_valid)
    point_biserial_corr.append(pb_corr)

# LASSO Regression
lasso = LassoCV(cv=5).fit(pores_valid, kappa_valid)
lasso_coefs = lasso.coef_

# Random Forest Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(pores_valid, kappa_valid)
rf_importances = rf.feature_importances_

# Permutation Feature Importance
perm_importances = permutation_importance(rf, pores_valid, kappa_valid, n_repeats=10, random_state=42)

# Results
print("T-Tests (stat, p-value):", t_tests)
print("Spearman Correlations:", spearman_corr)
print("Point-Biserial Correlations:", point_biserial_corr)
print("LASSO Coefficients:", lasso_coefs)
print("Random Forest Feature Importances:", rf_importances)
print("Permutation Importances:", perm_importances.importances_mean)