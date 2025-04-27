import time
import sys
import jax
from flax import nnx
import optax
import numpy as np
from flax import nnx
from mpi4py import MPI
import jax.numpy as jnp

from modules.training_utils import (
    data_loader,
    choose_schedule, accumulate_gradients,distribute_multicore, 
    mpi_allreduce_gradients, update_curves, log_training_progress,
    plot_update_learning_curves, compute_loss
)   

from modules.params_utils import save_params

from models.model_utils import predict, plot_example
from models.ensembles import ensemble


def train_model(
    exp_name, model_name, 
    dataset_al, dataset_train, dataset_val, 
    model_real,
    stop_perc,
    loss_type, beta_loss,
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

    dataset_train_local, dataset_val_local, sub_comm, local_rank, model_id = distribute_multicore(dataset_train, dataset_val, model_real, size, rank, comm)
    
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

            return compute_loss(loss_type, kappa_pred, kappas, kappa_var, beta_loss) # =0.6 - 0.2 * epoch / epochs 

        # Compute the loss and gradients
        #loss, grads = nnx.value_and_grad(loss_fn)(model)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux =True)(model)

        return loss, grads, aux

    def validate(dataset, batch_size, comm, sub_comm, valid_size):

        if isinstance(model_real, ensemble):
            
            total_val_loss, total_error_perc, log_var_loss, squared_error, mse = valid_ensemble(comm, loss_type, dataset, model)
        
        else: 
            total_val_loss, total_error_perc, log_var_loss, squared_error, mse = 0.0, 0.0, 0.0, 0.0, 0.0

            for val_pores, val_kappas in data_loader(*dataset, batch_size=batch_size):
                kappa_val, kappa_var = predict(model, val_pores)
                local_loss, local_aux = compute_loss(loss_type, kappa_val, val_kappas, kappa_var, beta_loss=0.5)
                local_log_var, local_squared_error, local_mse = local_aux
                local_error = jnp.sum(jnp.abs(kappa_val - val_kappas) * 100.0 / jnp.abs(val_kappas))
                total_val_loss += sub_comm.allreduce(local_loss, op=MPI.SUM)
                total_error_perc += sub_comm.allreduce(local_error, op=MPI.SUM)
                mse += sub_comm.allreduce(local_mse, op=MPI.SUM)
                log_var_loss += sub_comm.allreduce(local_log_var, op=MPI.SUM)
                squared_error += sub_comm.allreduce(local_squared_error, op=MPI.SUM)
        
        return total_val_loss / valid_size, total_error_perc / valid_size, log_var_loss / valid_size, squared_error / valid_size, mse / valid_size

    if rank == 0:
        print(f"Past epochs: {n_past_epoch}")
        sys.stdout.flush()
   
    metrics = {
        "loss": np.zeros(epochs),
        "val_loss": np.zeros(epochs),
        "mse": np.zeros(epochs),
        "val_mse": np.zeros(epochs),
        "val_fract_error": np.zeros(epochs),
        "valid_variance": np.zeros(epochs),
        "log-var-error": np.zeros(epochs),
        "squared-error": np.zeros(epochs),
        "log-var-val": np.zeros(epochs),
        "squared-val": np.zeros(epochs),
    }

    achieved = False
    val_min = 1e10

    for epoch in range(epochs):
        
        if dataset_al is not None and dataset_al.checkupdate(epoch, metrics["val_fract_error"], achieved):

            dataset_train = dataset_al.sample(model_real, model, comm=comm, rank=rank, epoch=epoch)

            dataset_train_local, dataset_val_local, _, _, _ = distribute_multicore(dataset_train, dataset_val, model_real, size, rank, comm)

            optimizer = nnx.Optimizer(model, optax.adam(lr_schedule))

            sys.stdout.flush()

        epoch_time = time.time()
        grads, total_loss, log_var_loss, squared_error, mse_loss = None, 0.0, 0.0, 0.0, 0.0

        for batch_n, batch_local in enumerate(data_loader(*dataset_train_local, batch_size=batch_size)):

            local_loss, local_grads, local_aux = loss_and_grads(batch_local, epoch, batch_n, rank)

            local_log_var, local_squared_error, mse_loss = local_aux

            # Reduce accross same-model cores
            new_local_loss = sub_comm.allreduce(local_loss, op=MPI.SUM)
            new_mse_loss = sub_comm.allreduce(mse_loss, op=MPI.SUM)
            new_local_log_var = sub_comm.allreduce(local_log_var, op=MPI.SUM)
            new_local_squared_error = sub_comm.allreduce(local_squared_error, op=MPI.SUM)
            new_local_grads = mpi_allreduce_gradients(local_grads, sub_comm)

            # Accumulate Loss and gradients
            total_loss += new_local_loss
            mse_loss += new_mse_loss
            log_var_loss += new_local_log_var
            squared_error += new_local_squared_error
            grads = accumulate_gradients(grads, new_local_grads)

        # Optimizer step
        optimizer.update(grads)

        # Aggregate training loss across samples
        train_loss = total_loss / dataset_train[0].shape[0]
        mse_loss = mse_loss / dataset_train[0].shape[0]
        log_var_loss = log_var_loss / dataset_train[0].shape[0]
        squared_error = squared_error / (dataset_train[0].shape[0])

        val_loss, fract_perc_loss_perc, log_var_val, squared_val, val_mse_loss = validate(dataset_val_local, batch_size, comm, sub_comm, dataset_val[0].shape[0])
        
        # Train
        metrics["loss"][epoch] = train_loss
        metrics["mse"][epoch] = mse_loss
        metrics["log-var-error"][epoch] = log_var_loss
        metrics["squared-error"][epoch] = squared_error

        # val
        metrics["val_loss"][epoch] = val_loss
        metrics["val_mse"][epoch] = val_mse_loss
        metrics["val_fract_error"][epoch] = fract_perc_loss_perc
        metrics["log-var-val"][epoch] =log_var_val
        metrics["squared-val"][epoch] = squared_val

        # Check min
        if (epoch+1)%100 == 0 and val_loss < val_min:
            val_min = val_loss
            save_parameters_wrap(exp_name, model_name, model_real, model, checkpointer, rank)

        # Log the training progress for each epoch
        log_progress(
            loss_type, model_real, model, model_id, rank, epoch, n_past_epoch, epochs,
            metrics, time.time() - epoch_time, comm, dataset_val, size, n_models)
        
        if rank ==0 and epoch % 50 == 0 and debug:
            plot_example(
                model, dataset_train[0], dataset_train[1], epoch
            )

        sys.stdout.flush()

        if not achieved and fract_perc_loss_perc <= stop_perc[0]:
            
            print(f"Achieved percentual loss on val set of {stop_perc[0]} in {epoch} epochs with {len(dataset_train_local[0])}")
            stop_perc.pop(0)
            if stop_perc == []:
                achieved = True

        if epoch % 50 == 0:
            jax.clear_caches()

    
    plot_update_learning_curves(exp_name, model_name, n_past_epoch, epoch, metrics)

    if metrics["val_loss"][-1] < val_min:
        save_parameters_wrap(exp_name, model_name, model_real, model, checkpointer, rank)

    return model_real, metrics["mse"][-1].item(), metrics["val_mse"][-1].item(), metrics['val_fract_error'][-1].item()



def valid_ensemble(comm, loss_type, dataset_val, model):
    # Debug output: Check a few predictions
    
    comm.Barrier()  # Synchronize all ranks

    # Now, proceed with the prediction on each rank
    val_pores, val_kappas = dataset_val 
    kappa_pred_local, kappa_var_local = predict(model, val_pores)

    local_loss, local_aux = compute_loss(loss_type, kappa_pred_local, val_kappas, kappa_var_local, beta_loss=0.5)

    local_log, local_sq, local_mse = local_aux

    # Gather data from all ranks on rank 0
    total_loss = comm.allreduce(local_loss, op=MPI.SUM)
    total_loss /= comm.Get_size()
    total_log_var = comm.allreduce(local_log, op=MPI.SUM)
    total_log_var /= comm.Get_size()
    total_squared_error = comm.allreduce(local_sq, op=MPI.SUM)
    total_squared_error /= comm.Get_size()
    total_mse = comm.allreduce(local_mse, op=MPI.SUM)
    total_mse /= comm.Get_size()

    kappa_pred_all = comm.allgather(kappa_pred_local)
   
    kappa_pred_stacked = jnp.stack(kappa_pred_all)
    kappa_pred_mean = jnp.mean(kappa_pred_stacked, axis=0)
    """print(f"Rank {comm.Get_rank()} - kappa_pred_all shape: {kappa_pred_local[0:4]}")
    if comm.Get_rank() == 0:
        print(f"Mean: {kappa_pred_mean[0:4]}, Real {val_kappas[0:4]}")"""
    fract_val_error = jnp.sum(jnp.abs(kappa_pred_mean - val_kappas) * 100.0 / jnp.abs(val_kappas))

    return total_loss, fract_val_error, total_log_var, total_squared_error, total_mse

def save_parameters_wrap(exp_name, model_name, model_real, model, checkpointer, rank):

    if isinstance(model_real, ensemble):
        model_name = model_name + f"/model_{rank}"
        
        save_params(exp_name, model_name, model, checkpointer[rank])
        checkpointer[rank].wait_until_finished()  # <-- important!

    elif not isinstance(model_real, ensemble) and rank == 0:
        save_params(exp_name, model_name, model, checkpointer)
        checkpointer.wait_until_finished()  # <-- important!

def log_progress(loss_type, model_real, model, model_id, rank, epoch, n_past_epoch, epochs, metrics, epoch_time, comm, dataset_val, size, n_models):

    # Train Set
    train_loss = metrics["loss"][epoch]
    mse = metrics["mse"][epoch]
    log_var_loss = metrics["log-var-error"][epoch]
    squared_error = metrics["squared-error"][epoch]

    # val Set
    val_loss = metrics["val_loss"][epoch]
    fract_perc_loss = metrics["val_fract_error"][epoch]
    val_mse = metrics["val_mse"][epoch]
    log_var_val = metrics["log-var-val"][epoch]
    squared_val = metrics["squared-val"][epoch]

    if isinstance(model_real, ensemble):
            log_training_progress(loss_type, model_id, epoch, n_past_epoch, epochs,
                         train_loss, val_loss, fract_perc_loss, epoch_time,
                          mse, val_mse,
                          log_var_loss, log_var_val,
                          squared_error, squared_val)
            
    elif rank % (size // n_models) == 0:
        log_training_progress(loss_type, model_id, epoch, n_past_epoch, epochs,
                         train_loss, val_loss, fract_perc_loss, epoch_time,
                          mse, val_mse,
                          log_var_loss, log_var_val,
                          squared_error, squared_val)

    if isinstance(model_real, ensemble):
        pass
        #avg_val_loss, total_loss_perc = valid_ensemble(epoch, comm, dataset_val, model)