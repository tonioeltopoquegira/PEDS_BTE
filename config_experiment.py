basic_1000_train = {
    "exp_name": "train_1000",
    "seed": 42,

    # Run
    "training": False,
    "optimization": True,

    # Data
    "train_size": 1000, # change to train, test, validation
    "test_size": 100,

    # Active Learning
    "al": False,
    "N": 400,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500],
     
    # Training
    "epochs": 1000,
    "batch_size": 250,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "ga"
}


basic_1000_train_w1000test = {
    "exp_name": "train_1000_w1000test",
    "seed": 43,

    # Run
    "training": True,
    "optimization": True,

    # Data
    "train_size": 1000, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": False,
    "N": 400,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500],
     
    # Training
    "epochs": 1000,
    "batch_size": 250,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "ga"
}