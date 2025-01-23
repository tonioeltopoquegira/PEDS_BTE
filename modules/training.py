import time
import sys
from mpi4py import MPI

import jax.numpy as jnp
import numpy as np

import optax
from flax import nnx

from modules.params_utils import save_params
from modules.training_utils import data_loader, print_generated, update_and_check_grads, clip_gradients, plot_learning_curves, choose_schedule, accumulate_gradients, distribute_dataset, mpi_allreduce_gradients, create_folders, final_validation
from models.peds import PEDS



def train_model(model_name,
                dataset_train, dataset_valid, 
                model,
                learn_rate_max, learn_rate_min, schedule,  epochs, batch_size,
                checkpointer): 

    
    create_folders(model_name)

    try:
        curves = np.load(f"data/results/{model_name}/training_curves.npz", allow_pickle=True)
        n_past_epoch = len(curves['epoch_times'])+1
    except Exception as e:
        n_past_epoch = 0
    print(f"Past epochs: {n_past_epoch}")

    # Scheduler optimizer
    lr_schedule = choose_schedule(schedule, learn_rate_min, learn_rate_max, epochs)
    optimizer = nnx.Optimizer(model, optax.adam(lr_schedule))
    
    def train_step(pores, conductivities, kappas, epoch, batch, rank):
        
        if isinstance(model, PEDS):
            def loss_fn(model):
                kappa_pred, conductivity_res = model(pores, conductivities) # change here
                if (epoch+1) % 50 == 0 and batch == 0 and rank == 0:
                    print_generated(model, conductivities, conductivity_res, epoch+1+n_past_epoch, model_name, kappa_pred, kappas) # change here
                residuals = kappa_pred - kappas
                loss = jnp.sum(residuals**2)

                return loss
        else:
            def loss_fn(model):
                pores_reshaped = jnp.reshape(pores, (pores.shape[0], 25))
                kappa_pred = model(pores_reshaped)
                kappa_pred = jnp.squeeze(kappa_pred, -1)
                residuals = kappa_pred - kappas
                loss = jnp.sum(residuals**2)
                return loss
            
        # Compute loss and gradients
        loss, grads = nnx.value_and_grad(loss_fn)(model)

        return loss, grads

    # OCCHIO A QUESTO... COMM.ALL_REDUCE NON STA VENENDO USATO CORRETTAMENTE PENSO... OCCHIO ALLA VALIDATION CHE ORA SEMBRA ANDARE 
    def valid_step(dataset, batch_size, comm, valid_size):

        # 2000, 5, 5 ---> 10, 200, 5, 5 ne avanzano 2 dopo le prime 8 diversamente da
        # 8000, 5, 5 ---> 40, 200, 5, 5

        batch_size = 125
        total_val_loss = 0.0
        total_error_perc  = 0.0

        for en, val_batch in enumerate(data_loader(*dataset, batch_size=batch_size)):
            val_pores, val_conductivities, val_kappas = val_batch

            if isinstance(model, PEDS):
                kappa_val, _ = model(val_pores, val_conductivities) # here
            else:
                val_pores_reshaped = jnp.reshape(val_pores, (val_pores.shape[0], 25))
                kappa_val = model(val_pores_reshaped)
                kappa_val = jnp.squeeze(kappa_val, -1)

            local_res = (kappa_val - val_kappas)**2
            local_error = jnp.abs(kappa_val - val_kappas)*100.0/jnp.abs(val_kappas) 

            # To be adapted to 
            total_val_loss += comm.allreduce(jnp.sum(local_res), op=MPI.SUM)
            total_error_perc += comm.allreduce(jnp.sum(local_error), op=MPI.SUM)
        
        total_val_loss = total_val_loss / valid_size
        total_error_perc = total_error_perc / valid_size
        
        return total_val_loss, total_error_perc

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Current process ID
    size = comm.Get_size()  # Total number of processes

    epoch_losses = np.zeros(epochs) 
    valid_losses = np.zeros(epochs)
    valid_perc_losses = np.zeros(epochs)
    epoch_times = np.zeros(epochs)

    # Shard training and validation datasets
    dataset_train_local = distribute_dataset(dataset_train, rank, size)
    dataset_valid_local = distribute_dataset(dataset_valid, rank, size)

    sys.stdout.flush()

    for epoch in range(epochs):

        epoch_time = time.time()
        grads = None
        total_loss = 0.0 

        for en, batch in enumerate(data_loader(*dataset_train_local, batch_size=batch_size)):

            pores_local, conductivities_local, kappas_local = batch
            
            # Compute loss and gradients locally
            local_loss, local_grads = train_step(pores_local, conductivities_local, kappas_local, epoch, en, rank)
            
            # Accumulate loss across ranks
            total_loss += comm.allreduce(local_loss, op=MPI.SUM)

            grads = accumulate_gradients(grads, mpi_allreduce_gradients(local_grads, comm))
        
        # Update of parameters!
        optimizer.update(grads)

        # Validation step
        avg_val_loss, total_loss_perc = valid_step(dataset_valid_local, batch_size, comm, valid_size = dataset_valid[0].shape[0])

        avg_loss = total_loss / dataset_train[0].shape[0]
        epoch_times[epoch] = time.time() - epoch_time
        epoch_losses[epoch] = avg_loss
        valid_losses[epoch] = avg_val_loss
        valid_perc_losses[epoch] = total_loss_perc

        if rank == 0 and (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Losses: [{avg_loss:.2f}] , Validation Losses: [{avg_val_loss:.2f}, {total_loss_perc:.2f}%], Epoch time: {time.time() - epoch_time:.2f}s")
        
        sys.stdout.flush() 

        if (epoch + 1) % 100 == 0 and rank == 0:
            try:
                curves = np.load(f"data/results/{model_name}/training_curves.npz", allow_pickle=True)
                
                # Concatenate only new data
                epoch_times_tot = np.concatenate([curves['epoch_times'][:n_past_epoch], epoch_times[:epoch]])
                epoch_losses_tot = np.concatenate([curves['epoch_losses'][:n_past_epoch], epoch_losses[:epoch]])
                valid_losses_tot = np.concatenate([curves['valid_losses'][:n_past_epoch:], valid_losses[:epoch]])
                valid_perc_losses_tot = np.concatenate([curves['valid_perc_losses'][:n_past_epoch:], valid_perc_losses[:epoch]])
                
                # Calculate total epochs
                epoch_tot = len(epoch_losses_tot)
                
                # Plot and save
                plot_learning_curves(epoch_times_tot, epoch_losses_tot, valid_losses_tot, valid_perc_losses_tot, schedule, model_name, epoch_tot, learn_rate_max, learn_rate_min)
                np.savez(
                    f"data/results/{model_name}/training_curves.npz", 
                    epoch_times=epoch_times_tot, 
                    epoch_losses=epoch_losses_tot,
                    valid_losses=valid_losses_tot, 
                    valid_perc_losses=valid_perc_losses_tot,
                    allow_pickle=True
                )
            except Exception as e:
                print(f"No training curves file: {e}. Creating new one.")
                plot_learning_curves(epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, model_name, epoch, learn_rate_max, learn_rate_min)
                np.savez(
                    f"data/results/{model_name}/training_curves.npz", 
                    epoch_times=epoch_times, 
                    epoch_losses=epoch_losses,
                    valid_losses=valid_losses, 
                    valid_perc_losses=valid_perc_losses,
                    allow_pickle=True
                )
            
            save_params(model_name, model, checkpointer)



    if rank == 0:
        """curves = np.load(f"data/results/{model_name}/training_curves.npz", allow_pickle=True)

        if (epoch+1) == 100:
            n_past_epoch = len(curves['epoch_times'])
        
        # Concatenate only new data
        epoch_times_tot = np.concatenate([curves['epoch_times'], epoch_times[n_past_epoch:epoch]])
        epoch_losses_tot = np.concatenate([curves['epoch_losses'], epoch_losses[n_past_epoch:epoch]])
        valid_losses_tot = np.concatenate([curves['valid_losses'], valid_losses[n_past_epoch:epoch]])
        valid_perc_losses_tot = np.concatenate([curves['valid_perc_losses'], valid_perc_losses[n_past_epoch:epoch]])
        
        # Calculate total epochs
        epoch_tot = len(epoch_losses_tot)
        
        # Plot and save
        plot_learning_curves(epoch_times_tot, epoch_losses_tot, valid_losses_tot, valid_perc_losses_tot, schedule, model_name, epoch_tot, learn_rate_max, learn_rate_min)
        np.savez(
            f"data/results/{model_name}/training_curves.npz", 
            epoch_times=epoch_times_tot, 
            epoch_losses=epoch_losses_tot,
            valid_losses=valid_losses_tot, 
            valid_perc_losses=valid_perc_losses_tot,
            allow_pickle=True
        )"""
    
        save_params(model_name, model, checkpointer)

        final_validation(model, model_name, dataset_valid)
    








    