from flax import nnx
import jax.numpy as jnp
from modules.training_utils import choose_activation


class mlp(nnx.Module):

    def __init__(self, layer_sizes:list, activation:str, rngs:nnx.Rngs):
        
        super().__init__()

        dense_init = nnx.initializers.xavier_normal()
        bias_init = nnx.initializers.constant(0.0)
        if activation == "relu":
            bias_init = nnx.initializers.constant(1.0)
            

        self.layers = [
            nnx.Linear(i, o, kernel_init=dense_init, bias_init=bias_init, rngs=rngs) 
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        
        self.activation = choose_activation(activation)
    
    def __call__(self, x):
        for en, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)

        # remember to move the last layer out for MLP_baseline

        return x

