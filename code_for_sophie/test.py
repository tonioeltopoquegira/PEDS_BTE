import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import matplotlib.pyplot as plt
import os

# Import the model and utility functions
from models import mlp, load_model_parameters, load_data


def compute_metrics(predictions, targets):
   
    mse = jnp.mean((predictions - targets) ** 2)
    fractional_error = jnp.mean(jnp.abs(predictions - targets) / jnp.abs(targets))
    
    return {"mse": float(mse),  "fractional_error": float(fractional_error)}



def main():
    
    print("Initializing model...")

    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key) # Useless in your case --> NOT TRAINING
    
    model = mlp(
        layer_sizes=[25, 32, 64, 64, 1],
        activation="mixed" ,
        initialization="xavier",
        rngs=rngs
    )
    
    # Choose "expert"
    #checkpoint_path = "params/mlp_all/final_20250729_0806"  
    #checkpoint_path = "params/mlp_small_k/final_20251002_0308"
    checkpoint_path = "params/mlp_big_k/final_20251002_0655"
    
    print("Loading model parameters...")
    model = load_model_parameters(model, checkpoint_path)
    
    print("Loading data...")
    data_path = "data/high_fidelity_2_20000.npz"
    pores, kappas = load_data(data_path)
    
    
    n_test = min(15000, len(pores))  
    test_indices = jnp.arange(n_test)
    
    X_test = pores[test_indices]
    y_test = kappas[test_indices]
    
    print(f"Using {n_test} samples for testing")
    
   
    print("Predictions...")
    
    predictions = model(X_test, training=False)
    

    
    
    pred_flat = predictions.flatten()
    y_flat = y_test.flatten()
    
   
    metrics_all = compute_metrics(pred_flat, y_flat)
    print(f"Test Results (All Data):")
    print(f"  MSE: {metrics_all['mse']:.6f}")
    print(f"  Fractional Error: {metrics_all['fractional_error']:.6f} ({metrics_all['fractional_error']*100:.2f}%)")
    print(f"  Total samples: {len(y_flat)}")
    
    # Metrics for kappas < 20
    mask_low = y_flat < 20
    if jnp.sum(mask_low) > 0:
        pred_low = pred_flat[mask_low]
        y_low = y_flat[mask_low]
        metrics_low = compute_metrics(pred_low, y_low)
        print(f"\nTest Results (κ < 20):")
        print(f"  MSE: {metrics_low['mse']:.6f}")
        print(f"  Fractional Error: {metrics_low['fractional_error']:.6f} ({metrics_low['fractional_error']*100:.2f}%)")
        print(f"  Samples: {len(y_low)}")
    else:
        print(f"\nTest Results (κ < 20): No samples found")
        metrics_low = None
    
    # Metrics for kappas > 45
    mask_high = y_flat > 45
    if jnp.sum(mask_high) > 0:
        pred_high = pred_flat[mask_high]
        y_high = y_flat[mask_high]
        metrics_high = compute_metrics(pred_high, y_high)
        print(f"\nTest Results (κ > 45):")
        print(f"  MSE: {metrics_high['mse']:.6f}")
        print(f"  Fractional Error: {metrics_high['fractional_error']:.6f} ({metrics_high['fractional_error']*100:.2f}%)")
        print(f"  Samples: {len(y_high)}")
    else:
        print(f"\nTest Results (κ > 45): No samples found")
        metrics_high = None
    
    # Metrics for middle range (20 <= κ <= 45)
    mask_mid = (y_flat >= 20) & (y_flat <= 45)
    if jnp.sum(mask_mid) > 0:
        pred_mid = pred_flat[mask_mid]
        y_mid = y_flat[mask_mid]
        metrics_mid = compute_metrics(pred_mid, y_mid)
        print(f"\nTest Results (20 ≤ κ ≤ 45):")
        print(f"  MSE: {metrics_mid['mse']:.6f}")
        print(f"  Fractional Error: {metrics_mid['fractional_error']:.6f} ({metrics_mid['fractional_error']*100:.2f}%)")
        print(f"  Samples: {len(y_mid)}")
    else:
        print(f"\nTest Results (20 ≤ κ ≤ 45): No samples found")
        metrics_mid = None


if __name__ == "__main__":
    main()