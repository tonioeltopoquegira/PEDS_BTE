basic_1000_train = {
    "exp_name": "train_1000_NEW",
    "seed": 32,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

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
    "batch_size": 1000,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": False,
    "beta_start": 35.0, 
    "beta_increase": 1.25,
    "lr": 0.01,

}

basic_10000_train = {
    "exp_name": "basic_10000",
    "seed": 44,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 3000, # change to train, test, validation
    "test_size": 500,

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
    "exp_name": "train_100_NEW",
    "seed": 42,
    "stop_perc": [0.0],

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 100, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": False,
    "N": 400,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500],
     
    # Training
    "epochs": 300,
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
    "exp_name": "train_200_NEW",
    "seed": 42,
    "stop_perc": [0.0],

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 200, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": False,
    "N": 400,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500],
     
    # Training
    "epochs": 400,
    "batch_size": 200,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga"
}

dataeff_300_train = {
    "exp_name": "train_300_NEW",
    "seed": 0,
    "stop_perc": [0.0],

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 300, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": False,
    "N": 400,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500],
     
    # Training
    "epochs": 600,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga"
}


dataeff_500_train = {
    "exp_name": "train_500_NEW",
    "seed": 42,
    "stop_perc": [0.0],

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": False,
    "N": 400,
    "M": 800,
    "K": 100,
    "T": [100, 200, 300, 400, 500],
     
    # Training
    "epochs": 700,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga"
}


dataeff_2000_train = {
    "exp_name": "train_2000_NEW",
    "seed": 42,
    "stop_perc": [0.0],

    # Run
    "training": True,
    "optimization": False,

    # Data
    "train_size": 2000, # change to train, test, validation
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
    "loss_beta": 0.4,

    # Optimization
    "opt": "ga"
}


earlystop_100 = {
    "exp_name": "earlystop_100",
    "seed": 42,

    # Run
    "training": True,
    "optimization": True,
    "stop_perc": [9.7, 7.0, 5.5, 5.0, 4.1],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 40, # 40
    "M": 600,
    "K": 40, # 40
    "T": [100], # 100
     
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

earlystop_200 = {
    "exp_name": "earlystop",
    "seed": 0,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [9.7, 7.1, 5.5, 5.0, 4.1],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 80,
    "M": 800,
    "K": 40,
    "T": [200],
     
    # Training
    "epochs": 600,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}

earlystop_500 = {
    "exp_name": "earlystop",
    "seed": 0,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [9.7, 7.1, 5.5, 5.0, 4.12],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 80,
    "M": 800,
    "K": 80,
    "T": [200, 350],
     
    # Training
    "epochs": 2000,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic":False
}

earlystop_1000 = {
    "exp_name": "earlystop",
    "seed": 0,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [9.7, 7.1, 5.5, 5.0, 4.12],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 150,
    "M": 1500,
    "K": 200,
    "T": [200],
     
    # Training
    "epochs": 800,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}

earlystop_2000 = {
    "exp_name": "earlystop",
    "seed": 0,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [9.7, 7.1, 5.5, 5.0, 4.12],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 300, # 300
    "M": 5000,
    "K": 600, # 300
    "T": [250], # [250, 300]
     
    # Training
    "epochs": 600,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}


al_100 = { # successful
    "exp_name": "al_100",
    "seed": 0, 

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 30,
    "M": 600,
    "K": 35,
    "T": [100, 200], # --> 8.66
    # Training
    "epochs": 400,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}



al_200 = {
    "exp_name": "al_200",
    "seed": 0, 

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 60,
    "M": 8000,
    "K": 70,
    "T": [100, 200],
     
    # Training
    "epochs": 400,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}

al_300 = {
    "exp_name": "al_300",
    "seed": 0,

    # Run
    "training": False,
    "optimization": True,
    "stop_perc": [0.0],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 60,
    "M": 1200,
    "K": 120,
    "T": [100, 250],
     
    # Training
    "epochs": 400,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}

al_500 = {
    "exp_name": "al_500",
    "seed": 0,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 100,
    "M": 2000,
    "K": 200,
    "T": [100, 250],
     
    # Training
    "epochs": 500,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": False
}

al_1000 = {
    "exp_name": "al_1000",
    "seed": 33,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 200, # 200
    "M": 4000,
    "K": 400, # 400
    "T": [300, 450], # 300, 450 -> 3.84
     
    # Training
    "epochs": 1000,
    "batch_size": 350,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}


al_2000 = {
    "exp_name": "al_2000",
    "seed": 33,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": True,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 400, # 200
    "M": 8000,
    "K": 800, # 400
    "T": [300, 450], # 300, 450 -> 3.49
     
    # Training
    "epochs": 1000,
    "batch_size": 350,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}

splits_0_1 = {
    "exp_name": "splits_0_1",
    "seed": 33,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 1000, 
    "test_size": 1000,

    # Splits
    "splits": (0, 1), # (a, b)

    # Active Learning
    "al": False,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 200, # 200
    "M": 4000,
    "K": 400, # 400
    "T": [300, 450], # 300, 450 -> 3.84
     
    # Training
    "epochs": 600,
    "batch_size": 350,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True 
}

splits_0_2 = {
    "exp_name": "splits_0_2",
    "seed": 33,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 1000, 
    "test_size": 1000,

    # Splits
    "splits": (0, 2), # (a, b)

    # Active Learning
    "al": False,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 200, # 200
    "M": 4000,
    "K": 400, # 400
    "T": [300, 450], # 300, 450 -> 3.84
     
    # Training
    "epochs": 600,
    "batch_size": 350,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True 
}

splits_1_0 = {
    "exp_name": "splits_1_0",
    "seed": 33,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 1000, 
    "test_size": 1000,

    # Splits
    "splits": (1, 0), # (a, b)

    # Active Learning
    "al": False,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 200, # 200
    "M": 4000,
    "K": 400, # 400
    "T": [300, 450], # 300, 450 -> 3.84
     
    # Training
    "epochs": 600,
    "batch_size": 350,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True 
}

splits_1_2 = {
    "exp_name": "splits_1_2",
    "seed": 33,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 1000, 
    "test_size": 1000,

    # Splits
    "splits": (1, 2), # (a, b)

    # Active Learning
    "al": False,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 200, # 200
    "M": 4000,
    "K": 400, # 400
    "T": [300, 450], # 300, 450 -> 3.84
     
    # Training
    "epochs": 600,
    "batch_size": 350,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True 
}

splits_2_1 = {
    "exp_name": "splits_2_1",
    "seed": 33,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 1000, 
    "test_size": 1000,

    # Splits
    "splits": (2, 1), # (a, b)

    # Active Learning
    "al": False,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 200, # 200
    "M": 4000,
    "K": 400, # 400
    "T": [300, 450], # 300, 450 -> 3.84
     
    # Training
    "epochs": 600,
    "batch_size": 350,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True 
}

splits_2_0 = {
    "exp_name": "splits_2_0",
    "seed": 33,

    # Run
    "training": True,
    "optimization": False,
    "stop_perc": [0.0],

    # Data
    "train_size": 1000, 
    "test_size": 1000,

    # Splits
    "splits": (2, 0), # (a, b)

    # Active Learning
    "al": False,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 200, # 200
    "M": 4000,
    "K": 400, # 400
    "T": [300, 450], # 300, 450 -> 3.84
     
    # Training
    "epochs": 600,
    "batch_size": 350,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True 
}


opt = {
    "exp_name": "al_300",
    "seed": 0,

    # Run
    "training": False,
    "optimization": True,
    "stop_perc": [0.0],

    # Data
    "train_size": 500, # change to train, test, validation
    "test_size": 1000,

    # Active Learning
    "al": False,
    "dynamic_query": False,
    "convergence": 0.025, # 0.025
    "N": 60,
    "M": 1200,
    "K": 120,
    "T": [100, 250],
     
    # Training
    "epochs": 400,
    "batch_size": 125,
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",
    "loss_beta": 0.5,

    # Optimization
    "opt": "ga",
    "stochastic": True
}
