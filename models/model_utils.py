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
            learn_residual=kwargs["learn_residual"], 
            hidden_sizes=kwargs["hidden_sizes"], 
            activation=kwargs["activation"], 
            solver=kwargs["solver"],
            initialization=kwargs['initialization'],
            seed=seed
        )
    elif model_type == "MLP":

        rngs = nnx.Rngs(seed=seed)
        return mlp(
            layer_sizes=[25] + kwargs["hidden_sizes"] + [1],  # Assuming this maps correctly
            activation=kwargs["activation"],
            initialization=kwargs['initialization'], 
            rngs=rngs
        )
    
    elif model_type == "ENSEMBLE":
        
        models = [PEDS(resolution=kwargs["resolution"], 
            learn_residual=kwargs["learn_residual"], 
            hidden_sizes=kwargs["hidden_sizes"], 
            activation=kwargs["activation"], 
            solver=kwargs["solver"],
            initialization=kwargs['initialization'], seed=seed+_) for _ in range(kwargs["n_models"])]


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
        kappa_mean, conductivity_generated = model(pores, training)
        if training:
            if (kwargs.get('epoch', 0) + 1 + kwargs.get('n_past_epoch', 0)) in [1, 2, 4, 7, 11, 17, 26, 39, 58, 86, 130, 195, 293, 440, 660, 999] and kwargs.get('batch_n', 0) == 0 and kwargs.get('rank', 0) == 0:
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
       # Create the figure and axes for 3x3 subplots
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [1, 1]})

        # Define the range for the color map to ensure all plots use the same scale
        vmin = min(np.min(c) for c in conductivity_res_numpy)
        vmax = max(np.max(c) for c in conductivity_res_numpy)

        # Plot the conductivities and residuals in the subplots
        for i in range(3):
            # Base conductivity
            """axes[i, 0].imshow(conductivities_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, 0].set_title('Base Conductivity')
            if i == 1:
                axes[i, 0].set_ylabel('y direction')"""

            # Generated conductivity
            im2 = axes[i, 1].imshow(conductivity_res_numpy[i], cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, 1].set_title('Generated / Final Conductivity')

            # Add text annotations for predicted vs. target values
            fig.text(
                0.90, 0.80 - i * 0.20,  # Adjust position based on row
                f"Predicted: {kappa_predicted_n[i]:.2f}\nTarget: {kappa_target[i]:.2f}",
                fontsize=10, color='black', ha='left', va='center'
            )

            # Create a single colorbar for all subplots (linked to final conductivity)
        fig.colorbar(im2, ax=axes[:3, :], orientation='vertical', label='Conductivity')

    
    # Save the figure
    plt.savefig(f"experiments/{exp_name}/figures/peds_evolution/{model_name}/conductivities_epoch_{epoch}.png")
    plt.close()