import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.nn as jnn

from utils import choose_nonlinearity

class Generator(nn.Module):

    hidden_sizes: list
    activations: list
    initializer: str

    def setup(self):

        # Initializers
        if self.initializer[0] == 1:
            dense_init = nn.initializers.xavier_uniform()

        elif self.initializer[0] == 2:
            dense_init = nn.initializers.normal(0.1)
        
        elif self.initializer[0] == 0:
            dense_init = nn.initializers.zeros_init()
        
        bias_init = nn.initializers.constant(self.initializer[0])
        
        # Define Layers from input_sizes
        self.layers = [nn.Dense(o, kernel_init=dense_init, bias_init=bias_init) for o in self.hidden_sizes[1:]]
    
        # Nonlinearity of choice
        self.nonlinearity_fn = [choose_nonlinearity(nonlin) for nonlin in self.activations]


    def __call__(self, X):
        
        for i, layer in enumerate(self.layers):
            X = self.nonlinearity_fn[i](layer(X))
        
        return X