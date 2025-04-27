import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
from flax import nnx

from models.peds import PEDS
from models.mlp import mlp
from models.ensembles import ensemble

from solvers.low_fidelity_solvers.base_conductivity_grid_converter import conductivity_original_wrapper

def select_model(seed, model_type, **kwargs):

    if model_type == "PEDS":
        return PEDS(
            resolution=kwargs["resolution"], 
            adapt_weights = kwargs["adapt_weights"],
            learn_residual=kwargs["learn_residual"], 
            hidden_sizes=kwargs["hidden_sizes"], 
            activation=kwargs["activation"], 
            solver=kwargs["solver"],
            initialization=kwargs['initialization'],
            uq_method=kwargs['uq_method'],
            n_modes=kwargs['n_modes'],
            hidden_sizes_uq=kwargs['hidden_sizes_uq'],
            seed=seed
        )
    elif model_type == "MLP":

        rng = jax.random.PRNGKey(seed)
        key = nnx.Rngs({'params': rng})

        return mlp(
            layer_sizes=[25] + kwargs["hidden_sizes"] + [1],  # Assuming this maps correctly
            activation=kwargs["activation"],
            initialization=kwargs['initialization'], 
            rngs=key
        )
    
    elif model_type == "ENSEMBLE":
        
        models = [PEDS(
            resolution=kwargs["resolution"], 
            adapt_weights = kwargs["adapt_weights"],
            learn_residual=kwargs["learn_residual"], 
            hidden_sizes=kwargs["hidden_sizes"], 
            activation=kwargs["activation"], 
            solver=kwargs["solver"],
            initialization=kwargs['initialization'],
            uq_method=kwargs['uq_method'],
            n_modes=kwargs['n_modes'],
            hidden_sizes_uq=kwargs['hidden_sizes_uq'],
            seed=seed+_)
            for _ in range(kwargs["n_models"])]

        return ensemble(
            models = models,
            n_models=kwargs["n_models"],  # Default to 2 if not specified
            uq_method=kwargs['uq_method']
        )
    
    elif model_type == "ENSEMBLE_MLP":

        models = [
            mlp(
            layer_sizes=[25] + kwargs["hidden_sizes"] + [1],  # Assuming this maps correctly
            activation=kwargs["activation"],
            initialization=kwargs['initialization'], 
            rngs=nnx.Rngs({'params': jax.random.PRNGKey(seed + _) })
        ) for _ in range(kwargs["n_models"])
        ]

        return ensemble(
            models = models,
            n_models=kwargs["n_models"],  # Default to 2 if not specified
        )


def predict(model, pores, training=False, **kwargs):

    conductivity_generated = None
    kappa_var = None

    if isinstance(model, mlp):
       
        pores_reshaped = jnp.reshape(pores, (pores.shape[0], 25))
        kappa_mean = jnp.squeeze(model(pores_reshaped, True), -1)
    
    if isinstance(model, PEDS):
        kappa_mean, kappa_var, conductivity_generated = model(pores, training)
        if training:
            if ((kwargs.get('epoch', 0) + 1 + kwargs.get('n_past_epoch', 0)) % 25 == 0 or kwargs.get('epoch', 0) ==0) and kwargs.get('batch_n', 0) == 0 and kwargs.get('rank', 0) == 0:
                plot_peds(model, pores, conductivity_res = conductivity_generated, model_name=kwargs.get('model_name'), exp_name=kwargs.get('exp_name'), epoch=kwargs.get('epoch') + 1 + kwargs.get('n_past_epoch'), kappa_predicted=kappa_mean, kappa_target=kwargs.get('kappas'))
    
    if isinstance(model, ensemble):
        
        kappa_mean, kappa_var = model(pores, training)

    return kappa_mean, kappa_var



def plot_peds(model, pores, conductivity_res, epoch, model_name, exp_name, kappa_predicted, kappa_target):
    # Ensure the arrays are NumPy arrays and detach from the computation graph
    conductivities_numpy = np.array(conductivity_original_wrapper(pores[:3], conductivity_res.shape[-1]))
    conductivity_res_numpy = [np.asarray(jax.lax.stop_gradient(r)) for r in conductivity_res[:3]]
    kappa_predicted_n = [np.asarray(jax.lax.stop_gradient(r)) for r in kappa_predicted[:3]]

    if model.learn_residual:

        # Create the figure and axes for 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15), gridspec_kw={"width_ratios": [1, 1, 1]})

        # Define the range for the color map to ensure all plots use the same scale
        vmin = min(np.min(c) for c in conductivities_numpy + conductivity_res_numpy)
        vmax = max(np.max(c) for c in conductivities_numpy + conductivity_res_numpy)

        # Plot the conductivities and residuals in the subplots
        for i in range(3):
            # Base conductivity
            axes[i, 0].imshow(conductivities_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, 0].set_title('Base Conductivity')

            # Generated conductivity
            axes[i, 1].imshow(conductivity_res_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, 1].set_title('Generated Conductivity')

            # Final conductivity (sum of base and residual)
            im3 = axes[i, 2].imshow(conductivity_res_numpy[i] + conductivities_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, 2].set_title('Final Conductivity')

            # Add text annotations for predicted vs. target values
            fig.text(
                0.90, 0.80 - i * 0.20,  # Adjust position based on row
                f"Predicted: {kappa_predicted_n[i]:.2f}\nTarget: {kappa_target[i]:.2f}",
                fontsize=10, color='black', ha='left', va='center'
            )

        # Create a single colorbar for all subplots (linked to final conductivity)
        fig.colorbar(im3, ax=axes[:, :3], orientation='horizontal', label='Conductivity')

    else:
       
        fig, axes = plt.subplots(3, 1, figsize=(5, 10), constrained_layout=True)

        # Set global vmin and vmax
        vmin = min(np.min(c) for c in conductivity_res_numpy)
        vmax = max(np.max(c) for c in conductivity_res_numpy)

        # Create a placeholder for the last image handle (used for colorbar)
        im = None

        for i in range(3):
            im = axes[i].imshow(conductivity_res_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Final Conductivity {i+1}')
            axes[i].axis('off')

        # Add a colorbar that applies to all subplots
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.04)
        cbar.set_label('Conductivity')

    
    # Save the figure
    plt.savefig(f"experiments/{exp_name}/figures/peds_evolution/{model_name}/conductivities_epoch_{epoch}.png")
    plt.close()


def plot_example(model, pores, k_true, epoch):

    kappa_pred, kappa_var = predict(model, pores)

    if kappa_var is not None and len(kappa_var.shape) > 1:
        kappa_var = jnp.exp(kappa_var.transpose((1, 0)))

    print(f"Real {k_true[13:16]}, Pred {kappa_pred[13:16]}, Var {kappa_var[13:16]}")


    plt.figure(figsize=(8, 5))
    plt.scatter(k_true, kappa_var, alpha=0.6, edgecolor='k')

    plt.xlabel('Predicted kappa (mean)')
    plt.ylabel('Predicted variance')
    plt.title('Predicted Variance vs. Predicted kappa')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"experiments/earlystop/figures/variance_vs_kappa_{epoch}.png")
    plt.close()

    # Compute standard deviation from variance
    std_dev = np.sqrt(kappa_var)  # shape (N,)

    # Get indices of 10 most uncertain predictions
    topk_idx = np.argsort(-std_dev)[:5]

    # Print details
    print("Top 5 most uncertain predictions:")
    for i, idx in enumerate(topk_idx):
        print(f"{i+1:2d}: μ = {kappa_pred[idx]:.4f}, σ = {std_dev[idx]:.4f}, GT = {k_true[idx]:.4f}, pores = {pores[idx]}")  # Adjusted to show pores

