m1 = {
    "model_name": "peds",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "relu", 
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
    "activation": "relu",
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 5
}



