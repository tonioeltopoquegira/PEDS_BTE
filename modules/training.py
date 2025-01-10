
import time
import os
import sys
from mpi4py import MPI

import jax
import jax.numpy as jnp
import numpy as np

import optax
from flax import nnx

from modules.params_utils import save_params
from modules.training_utils import data_loader, print_generated, update_and_check_grads, clip_gradients, plot_learning_curves, choose_schedule, accumulate_gradients, distribute_dataset, mpi_allreduce_gradients


def train_model(model_name,
                dataset_train, dataset_valid, 
                generator, 
                lowfidsolver, 
                learn_rate_max, learn_rate_min, schedule,  epochs, batch_size,
                checkpointer): 

    

    os.makedirs(f"figures/{model_name}", exist_ok=True)
    os.makedirs(f"figures/{model_name}/training_evolution", exist_ok=True)

    # Scheduler optimizer
    lr_schedule = choose_schedule(schedule, learn_rate_min, learn_rate_max, epochs)
    optimizer = nnx.Optimizer(generator, optax.adam(lr_schedule))


    def train_step(generator, lowfidsolver, pores, conductivities, kappas):

        def loss_fn(generator):
            kappa_pred, conductivity_res = predict(generator, lowfidsolver, pores, conductivities)
            residuals = kappa_pred - kappas
            return jnp.sum(residuals**2)

        loss, grads = nnx.value_and_grad(loss_fn)(generator)
        return loss, grads
    
    print("Training...")

    # model name for params and figures

    epoch_losses = np.zeros(epochs) # 
    valid_losses = np.zeros(epochs)
    valid_perc_losses = np.zeros(epochs)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Current process ID
    size = comm.Get_size()  # Total number of processes

    print(f"Process {rank} of {size} initialized")

    # Shard training and validation datasets
    dataset_train_local = distribute_dataset(dataset_train, rank, size)
    dataset_valid_local = distribute_dataset(dataset_valid, rank, size)

    print(dataset_train_local[0].shape)

    sys.stdout.flush()

    for epoch in range(epochs):

        epoch_time = time.time()

        grads = None
        total_loss = 0.0  # Initialize total loss for the epoch

        for en, batch in enumerate(data_loader(*dataset_train_local, batch_size=batch_size)):
            pores_local, conductivities_local, kappas_local = batch
            
            # Compute loss and gradients locally
            local_loss, local_grads = train_step(generator, lowfidsolver, pores_local, conductivities_local, kappas_local)

            #print(f"Batch {en} done for rank {rank}")
            
            # Accumulate loss across ranks
            total_loss += comm.allreduce(local_loss, op=MPI.SUM)

            grads = accumulate_gradients(grads, mpi_allreduce_gradients(local_grads, comm))
        

        avg_loss = total_loss / dataset_train[0].shape[0]

        train_time = time.time()

        avg_val_loss, total_loss_perc = valid(dataset_valid_local, batch_size, generator, lowfidsolver, comm, valid_size = dataset_valid[0].shape[0])

        sys.stdout.flush()
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.2f}, Validation Losses: [{avg_val_loss:.2f}, {total_loss_perc:.2f}%], Epoch time: {time.time() - epoch_time:.2f}s")
            print(f"Backprop part: {train_time - epoch_time}, Valid {time.time()-train_time}")


        epoch_losses[epoch] = avg_loss
        #avg_val_loss = jnp.sum(avg_val_loss)
        #total_loss_perc 
        valid_losses[epoch] = avg_val_loss
        valid_perc_losses[epoch] = total_loss_perc


        # grads = clip_gradients(grads, clip_value=0.99)
        
        # Calculate validation loss HERE 

        optimizer.update(grads)

        if epoch % 10 == 0 and rank==0:
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


# OCCHIO A QUESTO... ALL_REDUCE NON STA VENENDO USATO CORRETTAMENTE PENSO... OCCHIO ALLA VALIDATION CHE ORA SEMBRA ANDARE
def valid(dataset, batch_size, generator, lowfidsolver, comm, valid_size):

    # 2000, 5, 5 ---> 10, 200, 5, 5 ne avanzano 2 dopo le prime 8 diversamente da
    # 8000, 5, 5 ---> 40, 200, 5, 5

    batch_size = 125

    total_val_loss = 0.0
    total_error_perc  = 0.0
    for en, val_batch in enumerate(data_loader(*dataset, batch_size=batch_size)):
        val_pores, val_conductivities, val_kappas = val_batch
        kappa_val, _ = predict(generator, lowfidsolver, val_pores, val_conductivities)

        local_res = (kappa_val - val_kappas)**2
        local_error = jnp.abs(kappa_val - val_kappas)*100.0/jnp.abs(val_kappas) 

        # To be adapted to 
        total_val_loss += comm.allreduce(jnp.sum(local_res), op=MPI.SUM)
        total_error_perc += comm.allreduce(jnp.sum(local_error), op=MPI.SUM)
    
    total_val_loss = total_val_loss / valid_size
    total_error_perc = total_error_perc / valid_size
    

    return total_val_loss, total_error_perc



    
    





#size_square = int(10 * 1 / step_size)
#half_size_square = size_square // 2
#subgrid = jnp.ones((size_square, size_square)) * 1e-9
#indices = jnp.stack(jnp.meshgrid(jnp.arange(5), jnp.arange(5)), axis=-1).reshape(-1, 2)




