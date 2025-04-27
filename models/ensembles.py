from flax import nnx
import jax.numpy as jnp
    

class ensemble:
    def __init__(self, models:list,
                n_models: int, uq_method: int):
        
        self.models = models
        self.n_models = n_models
        self.uq_method = uq_method
    
    def __call__(self, x, training=False):
        outputs = [model(x, training)[0] for model in self.models]
        variances = [model(x, training)[1] for model in self.models]
        
        
        ensemble_mean = jnp.mean(jnp.stack(outputs), axis=0)

        if self.uq_method == 0:
            ensemble_var = jnp.var(jnp.stack(outputs), axis=0)
        
        elif self.uq_method == 1:
            var = [jnp.exp(var) for var in variances]
            var = jnp.mean(jnp.stack(var), axis=0).squeeze(-1)
            other = jnp.mean(jnp.stack(outputs)**2-ensemble_mean**2, axis=0)
           
            ensemble_var = var + other


       
        return ensemble_mean, ensemble_var