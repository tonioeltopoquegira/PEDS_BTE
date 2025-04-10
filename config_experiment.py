basic_1000_train = {
    "exp_name": "train_1000",
    "seed": 41,

    # Run
    "training": True,
    "valid": True, # change the validation to try different validations
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
    "batch_size": 250,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "ga",
    "kappas": [12.0, 15.0, 20.0, 30.0, 45.0, 60.0]
}
