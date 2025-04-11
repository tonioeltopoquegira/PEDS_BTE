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
    "model_name": "peds_mixed_smaller6",
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

# Baseline

mlpmod = {
    "model_name": "mlp6",
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
    "model_name": "pedsensemble",
    "model": "ENSEMBLE",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 10
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