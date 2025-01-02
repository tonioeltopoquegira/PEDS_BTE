import time
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from modules.params_utils import save_params
from modules.training_utils import data_loader, print_generated, update_and_check_grads, clip_gradients, plot_learning_curves, choose_schedule


def train_model(model_name,
                dataset_train, dataset_valid, 
                generator, 
                lowfidsolver, 
                learn_rate_max, learn_rate_min, schedule,  epochs, batch_size,
                checkpointer, ckpt_dir): 

    n_devices = jax.device_count()

    os.makedirs(f"figures/{model_name}", exist_ok=True)
    os.makedirs(f"figures/{model_name}/training_evolution", exist_ok=True)

    lr_schedule = choose_schedule(schedule, learn_rate_min, learn_rate_max, epochs)

    optimizer = nnx.Optimizer(generator, optax.adam(lr_schedule))


    def train_step(pores, conductivities, kappas, batch_n, epoch): # sharded pores and kappas
        
        def loss_fn(generator):
            
            kappa_pred, conductivity_res = predict(generator, lowfidsolver, pores, conductivities)
            residuals = (kappa_pred - kappas) 

            if batch_n == 0:
                #print(f"Kappas Pred:{kappa_pred[0].item()}, Kappa Target: {kappas[0]}")
                if epoch % 50 == 0:
                    print_generated(conductivities, conductivity_res, epoch, model_name, kappa_pred, kappas)

            return jnp.sum(residuals**2) # mean??

        loss, grads = nnx.value_and_grad(loss_fn)(generator)
        
        return loss, grads

    print("Training...")

    epoch_losses = np.zeros(epochs) # 
    valid_losses = np.zeros(epochs)

    for epoch in range(epochs):

        grads = None
        total_loss = 0.0  # Initialize total loss for the epoch

        batch_size = batch_size // n_devices

        for en, batch in enumerate(data_loader(*dataset_train, batch_size=batch_size)):
            
            pores, conductivities, kappas = batch

            loss, grads_new = train_step(pores, conductivities, kappas, en, epoch)
            
            total_loss += loss  # Add loss for the current batch
            
            grads = update_and_check_grads(grads, grads_new)
        
        avg_loss = total_loss / dataset_train[0].shape[0]



        total_val_loss = 0.0
        for en, val_batch in enumerate(data_loader(*dataset_valid, batch_size=batch_size)):
            val_pores, val_conductivities, val_kappas = val_batch
            kappa_val, _ = predict(generator, lowfidsolver, val_pores, val_conductivities)
            total_val_loss += jnp.sum((kappa_val - val_kappas)**2)
        
        avg_val_loss = total_val_loss / (dataset_valid[0]).shape[0]

        # Print the average loss at the end of each epoch
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}")

        epoch_losses[epoch] = avg_loss
        valid_losses[epoch] = avg_val_loss

        #grads = clip_gradients(grads, clip_value=0.99)
        
        # Calculate validation loss HERE 

        optimizer.update(grads)
        if epoch % 50 == 0:
            plot_learning_curves(epoch_losses, valid_losses, schedule, model_name, epoch)

    save_params(generator, checkpointer)
    
@nnx.jit
def predict(generator, lowfidsolver, pores, conductivities):

    conductivity_res = nnx.vmap(generator)(pores)
    #conductivity_res = nnx.vmap(generator, in_axes=0, out_axes=0)(conductivities)

    new_conductivity = conductivity_res+conductivities 

    new_conductivity = jnp.maximum(new_conductivity, 1e-5) # here we 

    """if (
        jnp.isinf(new_conductivity).any() 
        or jnp.isnan(new_conductivity).any() 
        or jnp.isclose(new_conductivity, 0).any()
    ):
        inf_indices = jnp.where(jnp.isinf(new_conductivity))
        nan_indices = jnp.where(jnp.isnan(new_conductivity))
        zero_indices = jnp.where(jnp.isclose(new_conductivity, 0))
        #print(f"Infinite values found at indices: {inf_indices}")
        #print(f"NaN values found at indices: {nan_indices}")
        print(f"Zero values found at indices: {zero_indices}")
        #raise ValueError("The computed conductivity contains infinity, NaN, or zero values.")"""
    
    kappa = lowfidsolver(new_conductivity) 

    """if jnp.isinf(kappa).any() or jnp.isnan(kappa).any():

        raise ValueError("The computed kappa contains infinity or nan values.")"""
    
    return kappa, conductivity_res



    
    





#size_square = int(10 * 1 / step_size)
#half_size_square = size_square // 2
#subgrid = jnp.ones((size_square, size_square)) * 1e-9
#indices = jnp.stack(jnp.meshgrid(jnp.arange(5), jnp.arange(5)), axis=-1).reshape(-1, 2)




