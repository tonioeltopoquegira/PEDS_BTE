from flax import nnx
import jax.numpy as jnp
from modules.training_utils import choose_activation


class mlp(nnx.Module):

    def __init__(self, layer_sizes:list, activation:str, rngs:nnx.Rngs, last_activation=False):
        
        super().__init__()

        dense_init = nnx.initializers.xavier_normal()
        bias_init = nnx.initializers.constant(0.0)
            
        self.last_activation = last_activation
        self.layers = [
            nnx.Linear(i, o, kernel_init=dense_init, bias_init=bias_init, rngs=rngs) 
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        
        self.activation = choose_activation(activation)
    
    def __call__(self, x):
        for en, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        x = self.layers[-1](x)

        if self.last_activation:
            x = self.activation(x)


        return x

