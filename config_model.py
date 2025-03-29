m1 = {
    "model_name": "ensemble_check",
    "model": "MLP",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "he",
    "n_models": 2
}

m2 = {
    "model_name": "PEDS_gauss",
    "model": "PEDS",
    "resolution": 20,
    "learn_residual": False,
    "hidden_sizes": [32, 64, 128],
    "activation": "relu",
    "solver": "gauss",
    "initialization": "he",
}



