from models.mlp import mlp
from models.peds import PEDS

def select_model(rngs, model_type, **kwargs):

    if model_type == "PEDS":
        return PEDS(
            resolution=kwargs["resolution"], 
            learn_residual=kwargs["learn_residual"], 
            hidden_sizes=kwargs["hidden_sizes"], 
            activation=kwargs["activation"], 
            solver=kwargs["solver"],
            init_min=kwargs['init_min'],
            initialization=kwargs['initialization'],
            final_init = kwargs['final_init']
        )
    elif model_type == "MLP":
        return mlp(
            layer_sizes=kwargs["hidden_sizes"],  # Assuming this maps correctly
            activation=kwargs["activation"],
            initialization=kwargs['initialization'], 
            final_init = False,
            rngs=rngs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")