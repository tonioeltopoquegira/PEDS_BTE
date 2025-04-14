# PEDS

peds_base = {

    "model_name": "peds_relu",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "relu", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1

}

peds_mixed = {
    "model_name": "peds1",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1
}

peds_small = {
    "model_name": "pedsmall4",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 25],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1
}

peds_fourier = {
    "model_name": "peds_fourier",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "fourier",
    "initialization": "xavier",
    "n_models": 1
}

# Baseline

mlpmod = {
    "model_name": "mlp1",
    "model": "MLP",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1
}

# Ensembles

peds_ens = {
    "model_name": "pedsensemble100",
    "model": "ENSEMBLE",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 4
}


mlp_ens = {
    "model_name": "mlpensemble",
    "model": "ENSEMBLE_MLP",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 10
}

peds_fourier = {

    "model_name": "peds_fourier_ens6",
    "model": "ENSEMBLE",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "fourier",
    "initialization": "xavier",
    "n_models": 4

}

peds_res_fourier = {

    "model_name": "peds_fourier",
    "model": "PEDS",
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "fourier",
    "initialization": "xavier",
    "n_models": 1

}

peds_fourier_ens = {

    "model_name": "peds_f_ens",
    "model": "ENSEMBLE",
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "fourier",
    "initialization": "xavier",
    "n_models": 4

}
