from flax import nnx
import jax.numpy as jnp
    

class ensemble:
    def __init__(self, models:list,
                n_models: int):
        
        self.models = models
        self.n_models = n_models
    
    def __call__(self, x, training=False):
        outputs = [model(x, training)[0] for model in self.models]
        ensemble_mean = jnp.mean(jnp.stack(outputs), axis=0)
        ensemble_var = jnp.var(jnp.stack(outputs), axis=0)
        return ensemble_mean, ensemble_var