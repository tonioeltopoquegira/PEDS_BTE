basic_1000_train = {
    "exp_name": "train_1000",
    "seed": 42,

    # Run
    "training": False,
    "optimization": True,
    "stop_perc": 0.0,

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


dataeff_100_train = {
    "exp_name": "train_100",
    "seed": 42,

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 100, # change to train, test, validation
    "test_size": 100,

    # Active Learning
    "al": False,
    "N": 400,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500],
     
    # Training
    "epochs": 1000,
    "batch_size": 100,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "ga"
}


dataeff_200_train = {
    "exp_name": "train_200",
    "seed": 42,

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 200, # change to train, test, validation
    "test_size": 100,

    # Active Learning
    "al": False,
    "N": 400,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500],
     
    # Training
    "epochs": 1000,
    "batch_size": 200,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "ga"
}

dataeff_500_train = {
    "exp_name": "train_500",
    "seed": 42,

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 500, # change to train, test, validation
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


dataeff_2000_train = {
    "exp_name": "train_2000",
    "seed": 42,

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 2000, # change to train, test, validation
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

active_learn = {
    "exp_name": "al_samedata",
    "seed": 42,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": 0.0,

    # Data
    "train_size": 1000, # change to train, test, validation
    "test_size": 100,

    # Active Learning
    "al": True,
    "N": 50,
    "M": 200,
    "K": 25,
    "T": [150, 300],
     
    # Training
    "epochs": 1000,
    "batch_size": 250,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "ga"
}

earlystop = {
    "exp_name": "earlystop",
    "seed": 42,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": 4.0,

    # Data
    "train_size": 1000, # change to train, test, validation
    "test_size": 100,

    # Active Learning
    "al": True,
    "dynamic_query": True,
    "convergence": 0.025,
    "N": 600,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500, 600],
     
    # Training
    "epochs": 10000,
    "batch_size": 250,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "ga"
}

attempt_fouier = {

    "exp_name": "code_fourier",
    "seed": 42,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": 0.0,

    # Data
    "train_size": 1000, # change to train, test, validation
    "test_size": 100,

    # Active Learning
    "al": False,
    "dynamic_query": False,
    "N": 200,
    "M": 400,
    "K": 50,
    "T": [100, 200, 300, 400, 500, 600],
     
    # Training
    "epochs": 1000,
    "batch_size": 250,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "ga"
}