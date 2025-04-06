from flax import nnx
import jax.numpy as jnp
    

class ensemble:
    def __init__(self, models:list,
                n_models: int):
        
        self.models = models
        self.n_models = n_models
    
    def __call__(self, x, training=False):
        outputs = [model(x, training)[0] for model in self.models]
        #print("Model 0 output ", outputs[0][:3])
        #print("Model 1 output ", outputs[1][:3])
        ensemble_mean = jnp.mean(jnp.stack(outputs), axis=0)
        #print("Mean ", ensemble_mean[:3])
        ensemble_var = jnp.var(jnp.stack(outputs), axis=0)
        #print(ensemble_var[:3])
        return ensemble_mean, ensemble_var