# Baseline

mlpmod = {
    # Model name and type
    "model_name": "mlp",
    "model": "MLP",

    # NN parameters
    "hidden_sizes": [32, 32, 32],
    "activation": "mixed", 
    "initialization": "xavier",

    # PEDS parameters
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "solver": "fourier",

    # UQ
    "uq_method": 0, 
    
    # UQ parameters
    "hidden_sizes_uq": [32, 32],
    "n_modes": 2,
    "n_models": 4
}


peds_fourier = {

    # Model name and type
    "model_name": "peds_fourier",
    "model": "PEDS",

    # NN parameters
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "initialization": "xavier",

    # PEDS parameters
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "solver": "fourier",

    # UQ
    "uq_method": 0, 
    
    # UQ parameters
    "hidden_sizes_uq": [32, 32], # 32 for best model
    "n_modes": 1,
    "n_models": 4

}

peds_gauss = {

    # Model name and type
    "model_name": "peds_gauss",
    "model": "PEDS",

    # NN parameters
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "initialization": "xavier",

    # PEDS parameters
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "solver": "gauss",

    # UQ
    "uq_method": 0, 
    
    # UQ parameters
    "hidden_sizes_uq": [32, 32], # 32 for best model
    "n_modes": 1,
    "n_models": 4

}


peds_f_ens = {

    # Model name and type
    "model_name": "peds_ens_fourier",
    "model": "ENSEMBLE",

    # NN parameters
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "initialization": "xavier",

    # PEDS parameters
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "solver": "fourier",

    # UQ
    "uq_method": 0, 
    
    # UQ parameters
    "hidden_sizes_uq": [32, 32],
    "n_modes": 2,
    "n_models": 4

}

peds_g_ens = {

    # Model name and type
    "model_name": "peds_ens_gauss",
    "model": "ENSEMBLE",

    # NN parameters
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "initialization": "xavier",

    # PEDS parameters
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "solver": "gauss",

    # UQ
    "uq_method": 0, # 0: basic ensemble, 1: logvar NN, 2: low order PCE
    
    # UQ parameters
    "hidden_sizes_uq": [32, 32],
    "n_modes": 2,
    "n_models": 4

}


peds_f_ens_uq1 = {

    # Model name and type
    "model_name": "peds_ens_fourier_uq1",
    "model": "ENSEMBLE",

    # NN parameters
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "initialization": "xavier",

    # PEDS parameters
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "solver": "fourier",

    # UQ
    "uq_method": 1, 
    
    # UQ parameters
    "hidden_sizes_uq": [32, 32],
    "n_modes": 2,
    "n_models": 4

}

peds_g_ens_uq1 = {

    # Model name and type
    "model_name": "peds_ens_gauss_uq1",
    "model": "ENSEMBLE",

    # NN parameters
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "initialization": "xavier",

    # PEDS parameters
    "resolution": 5,
    "adapt_weights": True,
    "learn_residual": True,
    "solver": "gauss",

    # UQ
    "uq_method": 1, 
    
    # UQ parameters
    "hidden_sizes_uq": [32, 32],
    "n_modes": 2,
    "n_models": 4

}