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

# Exp train 100

arch1 = {
    "model_name": "peds_arch14",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1
}

arch_direct = {
    "model_name": "peds_direct",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "mixed", 
    "solver": "direct",
    "initialization": "xavier",
    "n_models": 1
}

arch2 = {
    "model_name": "peds_arch2",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32, 32],
    "activation": "relu", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1
}

arch3 = {
    "model_name": "peds_arch3",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "relu", 
    "solver": "gauss",
    "initialization": "xavier",
    "n_models": 1
}

