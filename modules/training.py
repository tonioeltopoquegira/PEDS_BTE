
import time
import os
import sys

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
                checkpointer): 

    n_devices = jax.device_count()

    os.makedirs(f"figures/{model_name}", exist_ok=True)
    os.makedirs(f"figures/{model_name}/training_evolution", exist_ok=True)

    # Scheduler optimizer
    lr_schedule = choose_schedule(schedule, learn_rate_min, learn_rate_max, epochs)
    optimizer = nnx.Optimizer(generator, optax.adam(lr_schedule))


    def train_step(pores, conductivities, kappas, batch_n, epoch): # sharded pores and kappas
        
        def loss_fn(generator):
            
            kappa_pred, conductivity_res = predict(generator, lowfidsolver, pores, conductivities)
            residuals = (kappa_pred - kappas) 

            if batch_n == 0:
                #clear
                #print(f"Kappas Pred:{kappa_pred[0].item()}, Kappa Target: {kappas[0]}")
                if epoch % 1 == 0:
                    print_generated(conductivities, conductivity_res, epoch, model_name, kappa_pred, kappas)

            return jnp.sum(residuals**2)

        loss, grads = nnx.value_and_grad(loss_fn)(generator)
        
        return loss, grads


    @partial(
    pmap,
    axis_name='devices',
    static_broadcasted_argnums=(3, 4)  # Indices of `batch_n` and `epoch`
    )
    
    def parallel_train_step(pores, conductivities, kappas, batch_n, epoch):
        
        # `train_step` must return loss and gradients
        loss, grads = train_step(pores, conductivities, kappas, batch_n, epoch)
        #return loss, grads
        grads_tot = jax.tree_util.tree_map(lambda x: jax.lax.psum(x, axis_name='devices'), grads)
        loss_tot =  jax.tree_util.tree_map(lambda x: jax.lax.psum(x, axis_name='devices'), loss)
        return loss_tot, grads_tot
    
    #jax.pmap(parallel_train_step, axis_name="devices", static_broadcasted_argnums=(3,4))
    
    # Function to accumulate gradients
    def accumulate_gradients(total_grads, new_grads):
        if total_grads is None:
            return new_grads
        return jax.tree_util.tree_map(lambda x, y: x + y, total_grads, new_grads)

    
    print("Training...")

    epoch_losses = np.zeros(epochs) # 
    valid_losses = np.zeros(epochs)
    valid_perc_losses = np.zeros(epochs)

    for epoch in range(epochs):

        epoch_time = time.time()

        grads = None
        total_loss = 0.0  # Initialize total loss for the epoch
        

        #batch_size = batch_size // n_devices

        for en, batch in enumerate(data_loader(*dataset_train, batch_size=batch_size)):
            
            #pores_sharded, conductivities_sharded, kappas_sharded = batch

            #print("Pores before parallelization", pores_sharded.shape)

            #pores_sharded = pores.reshape(n_devices, -1, *pores.shape[1:])
            #conductivities_sharded = conductivities.reshape(n_devices, -1, *conductivities.shape[1:])
            #kappas_sharded = kappas.reshape(n_devices, -1, *kappas.shape[1:])
            
            # Perform parallel computation of loss and gradients
            #losses, new_grads = parallel_train_step(pores_sharded, conductivities_sharded, kappas_sharded, en, epoch)

            
            # Sum gradients across devices
            #grads = jax.tree_util.tree_map(lambda x: jax.lax.psum(x, axis_name='devices'), grads)
            
            # Accumulate gradients across batches
            #grads = accumulate_gradients(grads, new_grads)
        
            # Accumulate loss
            #total_loss += jnp.sum(losses)  # Sum losses across devices

            pores, conductivities, kappas = batch

            loss, grads_new = train_step(pores, conductivities, kappas, en, epoch)
            
            total_loss += loss  # Add loss for the current batch
            
            grads = update_and_check_grads(grads, grads_new)
        
        avg_loss = total_loss / dataset_train[0].shape[0]

        avg_val_loss, total_loss_perc = valid(dataset_valid, batch_size, generator, lowfidsolver)
        #avg_val_loss, total_loss_perc = 0.0, 0.0

        # Print the average loss at the end of each epoch
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.2f}, Validation Losses: [{avg_val_loss:.2f}, {total_loss_perc:.2f}%], Epoch time: {time.time() - epoch_time:.2f}s")

        epoch_losses[epoch] = avg_loss
        valid_losses[epoch] = avg_val_loss
        valid_perc_losses[epoch] = total_loss_perc


        # grads = clip_gradients(grads, clip_value=0.99)
        
        # Calculate validation loss HERE 

        optimizer.update(grads)

        if epoch % 10 == 0:
            plot_learning_curves(epoch_losses, valid_losses, valid_perc_losses, schedule, model_name, epoch, learn_rate_max, learn_rate_min)

    save_params(generator, checkpointer)
    

def predict(generator, lowfidsolver, pores, conductivities):

    
    # Process data through the generator (MLP)
    conductivity_res = nnx.jit(generator)(pores)
    #conductivity_res = nnx.vmap(generator, in_axes=1, out_axes=1)(conductivities)

    new_conductivity = conductivity_res+conductivities 

    new_conductivity = jnp.maximum(new_conductivity, 1e-5) # here we 

    kappa = lowfidsolver(new_conductivity) 

    
    return kappa, conductivity_res


def valid(dataset, batch_size, generator, lowfidsolver):

    total_val_loss = 0.0
    total_loss_perc = 0.0
    for en, val_batch in enumerate(data_loader(*dataset, batch_size=batch_size)):
        val_pores, val_conductivities, val_kappas = val_batch
        kappa_val, _ = predict(generator, lowfidsolver, val_pores, val_conductivities)
        total_val_loss += jnp.sum((kappa_val - val_kappas)**2)
        max_error_perc = jnp.max(jnp.abs(kappa_val - val_kappas)*100.0/val_kappas)
        total_loss_perc = jnp.maximum(total_loss_perc, max_error_perc)
    
    avg_val_loss = total_val_loss / (dataset[0]).shape[0]

    return avg_val_loss, total_loss_perc



    
    





#size_square = int(10 * 1 / step_size)
#half_size_square = size_square // 2
#subgrid = jnp.ones((size_square, size_square)) * 1e-9
#indices = jnp.stack(jnp.meshgrid(jnp.arange(5), jnp.arange(5)), axis=-1).reshape(-1, 2)




