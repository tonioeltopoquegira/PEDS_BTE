m1 = {
    "model_name": "peds",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1
}

m2 = {
    "model_name": "peds_ensemble",
    "model": "ENSEMBLE",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "mixed",
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 4
}

m3 = {
    "model_name": "peds_sanity",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "relu", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1
}

"""def train_step(model, batch_local, epoch, batch_n, rank, n_past_epoch, model_name, exp_name, model_id):
        pores, kappas = batch_local

        #print(f"[LOSS 1] Global rank {rank} -> Model ID {model_id}, Model object id: {id(model)}")

        def loss_fn(model):
            kappa_pred, kappa_var = predict(
                model, pores, training=True, epoch=epoch, 
                n_past_epoch=n_past_epoch, model_name=model_name, 
                exp_name=exp_name, kappas=kappas, batch_n=batch_n, rank=rank
            )
            residuals = kappa_pred - kappas
            return jnp.sum(residuals**2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)

        print(f"[LOSS 2] Global rank {rank} -> Model ID {model_id}, Model object id: {id(model)}: Loss {loss:.4f}")


       
        return loss, grads


"""