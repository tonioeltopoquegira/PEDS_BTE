basic_1000_train = {
    "exp_name": "train_1000",
    "seed": 42,

    # Run
    "training": True,
    "optimization": True,
    "stop_perc": [0.0],

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
    "batch_size": 500,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": False

}


dataeff_100_train = {
    "exp_name": "train_100",
    "seed": 42,
    "stop_perc": [0.0],

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
    "epochs": 600,
    "batch_size": 100,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic":False
}


dataeff_200_train = {
    "exp_name": "train_200",
    "seed": 42,
    "stop_perc": [0.0],

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
    "epochs": 600,
    "batch_size": 200,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga"
}

dataeff_500_train = {
    "exp_name": "train_500",
    "seed": 42,
    "stop_perc": [0.0],

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
    "epochs": 800,
    "batch_size": 250,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga"
}


dataeff_2000_train = {
    "exp_name": "train_2000",
    "seed": 43,
    "stop_perc": 0.0,

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
    "loss_beta": 0.4,

    # Optimization
    "opt": "ga"
}


earlystop = {
    "exp_name": "earlystop",
    "seed": 42,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [8.13, 6.91, 4.91, 4.12],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 100,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 40,
    "M": 200,
    "K": 25,
    "T": [400, 550, 700],
     
    # Training
    "epochs": 2000,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}

