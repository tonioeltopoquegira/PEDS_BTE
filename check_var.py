from models.peds import PEDS
from modules.params_utils import initialize_or_restore_params
from models.model_utils import select_model, predict
from flax import nnx
import jax.numpy as jnp

from config_model import peds_fourier_ens as model_config 
seed = nnx.Rngs(42)

# Select, create and initialize models
model = select_model(
    seed=42,    
    model_type=model_config["model"], 
    resolution=model_config["resolution"], 
    adapt_weights = model_config['adapt_weights'],
    learn_residual=model_config["learn_residual"], 
    hidden_sizes=model_config["hidden_sizes"], 
    activation=model_config["activation"],
    solver=model_config["solver"],
    initialization=model_config['initialization'],
    n_models = model_config['n_models']
)

# Params initializing or restoring
model, checkpointer = initialize_or_restore_params(model, model_config["model_name"], base_dir= "experiments/train_1000/weights", rank=0, seed=42) # check or do it deeper


designs = [jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]),
           jnp.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1]),
           jnp.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]),
           jnp.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]),
           jnp.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]),
           jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1]),
           jnp.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
           ]
for d in designs:
    d = d.reshape((1, 5, 5))
    print(f"Design {d} : {predict(model, d)}")