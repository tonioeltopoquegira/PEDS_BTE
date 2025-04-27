import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from models.peds import PEDS
from models.ensembles import ensemble
from models.model_utils import predict

def final_test(exp_name, model, model_name, dataset, mse_train, mse_val, perc_error):
   
    pores, kappa = dataset
    pores = pores.reshape((pores.shape[0], 25))
    kappa_pred, kappa_var = predict(model, pores, training=False)
    error = np.abs(kappa_pred - kappa) / np.abs(kappa_pred)
    error = error.mean().item() * 100.0

    with open(f"experiments/{exp_name}/results/errors.txt", "a") as f:
        f.write(f"{model_name}, {mse_train}, {mse_val}, {perc_error}, {error}\n")