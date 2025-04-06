e1 = {
    "exp_name": "coding",
    "seed": 10,

    # Run
    "training": True,
    "valid": True, # change the validation to try different validations
    "optimization": False,


    # Data
    "filename_data": "high_fidelity_2_20000.npz", # do we need this?
    "train_size": 1000, # change to train, test, validation
    "test_size": 100,
    "stratified": "all",

    # Active Learning
    "al": False,
    "N": 300,
    "M": 800,
    "K": 100,
    "T": 50,
     
    # Training
    "epochs": 1000,
    "batch_size": 200,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "grad",
    "kappas": [0.0, 10.0]
}

e2 = {
    "exp_name": "opt_coding",
    "seed": 42,

    # Run
    "training": False,
    "valid": False, # change the validation to try different validations
    "optimization": True,


    # Data
    "filename_data": "high_fidelity_2_20000.npz", # do we need this?
    "train_size": 1000, # change to train, test, validation
    "test_size": 100,
    "stratified": "all",

    # Active Learning
    "al": False,
    "N": 300,
    "M": 800,
    "K": 100,
    "T": 50,
     
    # Training
    "epochs": 1000,
    "batch_size": 200,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "grad",
    "kappas": [10.0, 15.0, 20.0, 30.]
}


e3 = {
    "exp_name": "saving_ensemble",
    "seed": 42,

    # Run
    "training": False,
    "valid": False, # change the validation to try different validations
    "optimization": True,


    # Data
    "filename_data": "high_fidelity_2_20000.npz", # do we need this?
    "train_size": 1000, # change to train, test, validation
    "test_size": 100,
    "stratified": "all",

    # Active Learning
    "al": False,
    "N": 300,
    "M": 800,
    "K": 100,
    "T": 50,
     
    # Training
    "epochs": 100,
    "batch_size": 200,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "grad",
    "kappas": [10.0]
}