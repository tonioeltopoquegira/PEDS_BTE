
from models.peds import PEDS
from models.mlp import mlp
from models.ensembles import ensemble

def select_model(rngs, model_type, **kwargs):

    if model_type == "PEDS":
        return PEDS(
            resolution=kwargs["resolution"], 
            learn_residual=kwargs["learn_residual"], 
            hidden_sizes=kwargs["hidden_sizes"], 
            activation=kwargs["activation"], 
            solver=kwargs["solver"],
            initialization=kwargs['initialization'],
        )
    elif model_type == "MLP":
        #
        return mlp(
            layer_sizes=[25] + kwargs["hidden_sizes"] + [1],  # Assuming this maps correctly
            activation=kwargs["activation"],
            initialization=kwargs['initialization'], 
            rngs=rngs
        )
    
    elif model_type == "ENSEMBLE":
        
        models = [PEDS(resolution=kwargs["resolution"], 
            learn_residual=kwargs["learn_residual"], 
            hidden_sizes=kwargs["hidden_sizes"], 
            activation=kwargs["activation"], 
            solver=kwargs["solver"],
            initialization=kwargs['initialization']) for _ in range(kwargs["n_models"])]


        return ensemble(
            models = models,
            n_models=kwargs["n_models"]  # Default to 2 if not specified
        )