config = {
    "experiment": "exp_test_code",
    "seed": 10,

    # Run
    "training": False,
    "valid": True, # change the validation to try different validations
    "optimization": False,

    # Data
    "filename_data": "high_fidelity_2_20000.npz", # do we need this?
    "train_size": 1000, # change to train, test, validation
    "total_size": 1100,
    "stratified": "small->small",

    # Model
    "model_name": "exp1_eff_srgt/PEDS_gauss5_arch6_mixed",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "mixed", # change activation to go automatically
    "solver": "gauss",
    "init_min": 1e-16,
    "initialization": "he",
    "reg": False,
    "final_init": False,
    "new_robust_check": True,

    # Training
    "epochs": 1000,
    "batch_size": 200,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "print_every": 10,

    # Results Training
    "mse_train": -1.0,
    "mse_test": -1.0,
    "perc_error": -1.0,

    # Optimization
    "opt": "ga",
    "kappas": [
        10.0, 12.63, 15.26, 17.89, 20.53, 23.16, 25.79, 28.42, 31.05, 33.68,
        36.32, 38.95, 41.58, 44.21, 46.84, 49.47, 52.11, 54.74, 57.37, 60.0,
        62.63, 65.26, 67.89, 70.53, 73.16, 75.79, 78.42, 81.05, 83.68, 86.32,
        88.95, 91.58, 94.21, 96.84, 99.47, 100.0
    ]
}
