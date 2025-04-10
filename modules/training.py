import time
import sys
import psutil
import os
import jax
from flax import nnx
import optax
import hashlib
from jax import tree_util
import numpy as np
from flax import nnx
from mpi4py import MPI
import jax.numpy as jnp

from modules.training_utils import (
    data_loader,
    choose_schedule, accumulate_gradients,distribute_multicore, 
    mpi_allreduce_gradients, update_curves, log_training_progress,
    plot_update_learning_curves 
)   

from modules.params_utils import save_params

from models.model_utils import predict
from models.ensembles import ensemble

def tree_norm(tree):
    """Compute L2 norm of a gradient PyTree"""
    leaves = tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves]))



def train_model(
    exp_name, model_name, 
    dataset_al, dataset_train, dataset_test, 
    model_real,
    learn_rate_max, learn_rate_min, schedule, epochs,
    batch_size, checkpointer,
    debug=False  # add a flag to toggle debug logging
):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_models = 1

    n_past_epoch = update_curves(model_name)
    lr_schedule = choose_schedule(rank, schedule, learn_rate_min, learn_rate_max, epochs, exp_name, n_past_epoch)

    dataset_train_local, dataset_test_local, sub_comm, local_rank, model_id = distribute_multicore(dataset_train, dataset_test, model_real, size, rank, comm)
    
    if isinstance(model_real, ensemble):
        model = model_real.models[model_id]
    else:
        model = model_real

    optimizer = nnx.Optimizer(model, optax.adam(lr_schedule))
        
    def loss_and_grads(batch_local, epoch, batch_n, rank):
        pores, kappas = batch_local

        def loss_fn(model):
            kappa_pred, kappa_var = predict(
                model, pores, training=True, epoch=epoch, 
                n_past_epoch=n_past_epoch, model_name=model_name, 
                exp_name=exp_name, kappas=kappas, batch_n=batch_n, rank=rank
            )
            residuals = kappa_pred - kappas
            return jnp.sum(residuals**2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)

        return loss, grads

    def validate(dataset, batch_size, comm, valid_size):
        total_val_loss, total_error_perc = 0.0, 0.0

        for val_pores, val_kappas in data_loader(*dataset, batch_size=batch_size):
            kappa_val, kappa_var = predict(model, val_pores)
            local_res = (kappa_val - val_kappas)**2
            local_error = jnp.abs(kappa_val - val_kappas) * 100.0 / jnp.abs(val_kappas)
            total_val_loss += comm.allreduce(jnp.sum(local_res), op=MPI.SUM)
            total_error_perc += comm.allreduce(jnp.sum(local_error), op=MPI.SUM)
        
        return total_val_loss / valid_size, total_error_perc / valid_size

    if rank == 0:
        print(f"Past epochs: {n_past_epoch}")
        sys.stdout.flush()
   
    epoch_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)
    valid_perc_losses = np.zeros(epochs)
    epoch_times = np.zeros(epochs)
    valid_variance = np.zeros(epochs)

    for epoch in range(epochs):

        if dataset_al is not None and dataset_al.checkupdate(epoch):
            dataset_train = dataset_al.sample(model, comm=comm, rank=rank)
            dataset_train_local, dataset_test_local, _, _, _ = distribute_multicore(dataset_train, dataset_test, model_real, size, rank, comm)

            sys.stdout.flush()

        epoch_time = time.time()
        grads, total_loss = None, 0.0

        for batch_n, batch_local in enumerate(data_loader(*dataset_train_local, batch_size=batch_size)):

            local_loss, local_grads = loss_and_grads(batch_local, epoch, batch_n, rank)

            # Reduce accross same-model cores
            new_local_loss = sub_comm.allreduce(local_loss, op=MPI.SUM)
            new_local_grads = mpi_allreduce_gradients(local_grads, sub_comm)

            # Accumulate Loss and gradients
            total_loss += new_local_loss
            grads = accumulate_gradients(grads, new_local_grads)

        # Optimizer step
        optimizer.update(grads)

        # Aggregate training loss across samples
        avg_loss = total_loss / dataset_train[0].shape[0]

        avg_val_loss, total_loss_perc = validate(dataset_test_local, batch_size, sub_comm, dataset_test[0].shape[0])

        # Log the training progress for each epoch
        if rank % (size // n_models) == 0:
            log_training_progress(model, model_id, rank, epoch, n_past_epoch, epochs, avg_loss, avg_val_loss, total_loss_perc, epoch_times)

        if isinstance(model_real, ensemble):
            avg_val_loss, total_loss_perc, avg_val_var = valid_ensemble(model_real, epoch, comm, dataset_test, model, rank)
            valid_variance[epoch] = avg_val_var
        
        epoch_times[epoch] = time.time() - epoch_time
        epoch_losses[epoch] = avg_loss
        valid_losses[epoch] = avg_val_loss
        valid_perc_losses[epoch] = total_loss_perc

            
        if epoch % 50 == 0:
            jax.clear_caches()

    
    # Final curves saving and return of final metrics.  
    if (rank % (size // n_models) == 0):
        plot_update_learning_curves(exp_name, model_name, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, valid_variance, schedule, learn_rate_max, learn_rate_min)
        if isinstance(model_real, ensemble):
            model_name = model_name + f"/model_{rank}"
            
            save_params(exp_name, model_name, model, checkpointer[rank])
            checkpointer[rank].wait_until_finished()  # <-- important!
        else:
            save_params(exp_name, model_name, model, checkpointer)
            checkpointer.wait_until_finished()  # <-- important!

        
    return model_real, avg_loss.item(), avg_val_loss.item(), total_loss_perc.item()



def valid_ensemble(model_real, epoch, comm, dataset_test, model, rank):
    # Debug output: Check a few predictions
    
    comm.Barrier()  # Synchronize all ranks


    # Now, proceed with the prediction on each rank
    val_pores, val_kappas = dataset_test 
    kappa_pred_local, kappa_var = predict(model, val_pores)

    # Gather data from all ranks on rank 0
    
    kappa_pred_all = comm.allgather(kappa_pred_local)

    kappa_pred_stacked = jnp.stack(kappa_pred_all)

    kappa_pred_mean = jnp.mean(kappa_pred_stacked, axis=0)
    kappa_pred_var = jnp.var(kappa_pred_stacked, axis=0)

    ens_valid_err = jnp.mean((kappa_pred_mean - val_kappas)**2)
    ens_valid_perc = jnp.mean(jnp.abs(kappa_pred_mean - val_kappas) * 100.0 / jnp.abs(val_kappas))
    ens_valid_var = jnp.mean(kappa_pred_var)

    if rank == 0 and epoch % 25 == 0:
        print(f"Ensemble : Epoch {epoch} | ValLoss: {ens_valid_err:.2f}, {ens_valid_perc:.2f}% | True kappas: {val_kappas[:3]}, Predicted: {kappa_pred_mean[:3]}, Variances: {kappa_pred_var[:3]}")

    return ens_valid_err, ens_valid_perc, ens_valid_var

