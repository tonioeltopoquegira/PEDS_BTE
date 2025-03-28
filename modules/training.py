import time
import sys
import jax
import optax
import numpy as np
from flax import nnx
from mpi4py import MPI
import jax.numpy as jnp

from modules.training_utils import (
    data_loader, print_generated, 
    choose_schedule, accumulate_gradients,distribute_dataset, 
    mpi_allreduce_gradients, update_curves, log_training_progress,
    curves_params
)   

from models.peds import PEDS
from models.ensembles import ensemble

def train_model(
    exp_name, model_name, 
    dataset_train, dataset_test, 
    model,
    learn_rate_max, learn_rate_min, schedule, epochs,
    batch_size, checkpointer
):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_past_epoch = update_curves(model_name)
    lr_schedule = choose_schedule(rank, schedule, learn_rate_min, learn_rate_max, epochs, exp_name, n_past_epoch)

    if isinstance(model, ensemble):
            num_models = model.n_models
            ranks_per_model = size // num_models
            model_id = rank // ranks_per_model
            model_real = model
            sub_comm = comm.Split(color=model_id, key=rank)
            model = model.models[model_id]
            checkpointer = checkpointer[model_id]
            optimizer = nnx.Optimizer(model, optax.adam(lr_schedule))
    else:
        optimizer = nnx.Optimizer(model, optax.adam(lr_schedule))
        sub_comm = comm

    def train_step(batch_local, epoch, batch_n, rank):
        pores, kappas = batch_local

        def loss_fn(model):
            if isinstance(model, PEDS):
                kappa_pred, conductivity_res = model(pores, True)
                if (epoch + 1 + n_past_epoch) in [1, 2, 4, 7, 11, 17, 26, 39, 58, 86, 130, 195, 293, 440, 660, 999] and batch_n == 0 and rank == 0:
                    print_generated(model, pores, conductivity_res, epoch + 1 + n_past_epoch, model_name, exp_name, kappa_pred, kappas)
            elif isinstance(model, ensemble):
                kappa_pred, _ = model(pores, True)
            else:
                pores_reshaped = jnp.reshape(pores, (pores.shape[0], 25))
                kappa_pred = jnp.squeeze(model(pores_reshaped, True), -1)
            residuals = kappa_pred - kappas
            return jnp.sum(residuals**2)

        return nnx.value_and_grad(loss_fn)(model)

    def valid_step(dataset, batch_size, comm, valid_size):
        total_val_loss, total_error_perc = 0.0, 0.0

        for val_pores, val_kappas in data_loader(*dataset, batch_size=batch_size):
            kappa_val = model(val_pores, True)[0] if isinstance(model, (PEDS, ensemble)) else jnp.squeeze(model(jnp.reshape(val_pores, (val_pores.shape[0], 25))), -1)
            local_res = (kappa_val - val_kappas)**2
            local_error = jnp.abs(kappa_val - val_kappas) * 100.0 / jnp.abs(val_kappas)
            total_val_loss += comm.allreduce(jnp.sum(local_res), op=MPI.SUM)
            total_error_perc += comm.allreduce(jnp.sum(local_error), op=MPI.SUM)
        
        return total_val_loss / valid_size, total_error_perc / valid_size


    if rank == 0:
        print(f"Past epochs: {n_past_epoch}")
        sys.stdout.flush()

    epoch_losses, valid_losses, valid_perc_losses, epoch_times = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)
    dataset_train_local, dataset_test_local = distribute_dataset(dataset_train, rank, size), distribute_dataset(dataset_test, rank, size)

    for epoch in range(epochs):
        epoch_time = time.time()
        grads, total_loss = None, 0.0

        for en, batch_local in enumerate(data_loader(*dataset_train_local, batch_size=batch_size)):
            local_loss, local_grads = train_step(batch_local, epoch, en, rank)
            total_loss += sub_comm.allreduce(local_loss, op=MPI.SUM)
            grads = accumulate_gradients(grads, mpi_allreduce_gradients(local_grads, sub_comm))

        optimizer.update(grads)

        # Aggregate training loss across models
        avg_loss = total_loss / dataset_train[0].shape[0]
        avg_loss = comm.allreduce(avg_loss, op=MPI.SUM) / model_real.n_models  # Ensure global mean

        # This shows that they are distinct models!
        #avg_loss = total_loss / dataset_train[0].shape[0]
       

        avg_val_loss, total_loss_perc = valid_step(dataset_test_local, batch_size, sub_comm, dataset_test[0].shape[0]) # this values are just for one model... sum between all 5 models (wait for them!)

        # Wait for all models and sum their validation losses
        avg_val_loss = comm.allreduce(avg_val_loss, op=MPI.SUM) / model_real.n_models
        total_loss_perc = comm.allreduce(total_loss_perc, op=MPI.SUM) / model_real.n_models


        
        epoch_times[epoch], epoch_losses[epoch], valid_losses[epoch], valid_perc_losses[epoch] = time.time() - epoch_time, avg_loss, avg_val_loss, total_loss_perc

        log_training_progress(model_real, rank, epoch, n_past_epoch, epochs, avg_loss, avg_val_loss, total_loss_perc, epoch_times)

        if epoch % 25 == 0:
            jax.clear_caches()

        if (epoch + 1) % 500 == 0 and rank == 0:
            curves_params(exp_name, model_name, model, checkpointer, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min)

    if isinstance(model_real, ensemble):
        if rank == 0:
            curves_params(exp_name, model_name, model, checkpointer, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min)
        return avg_loss.item(), avg_val_loss.item(), total_loss_perc.item()

          
    if rank == 0:

        curves_params(exp_name, model_name, model, checkpointer, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min)
        return avg_loss.item(), avg_val_loss.item(), total_loss_perc.item()





    