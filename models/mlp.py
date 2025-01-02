from flax import nnx
import jax.numpy as jnp
from modules.training_utils import hardtanh

class mlp(nnx.Module):

    def __init__(self, input_size:int, hidden_sizes:int, step_size:int, rngs:nnx.Rngs):
        
        super().__init__()

        dense_init = nnx.initializers.xavier_normal()
        bias_init = nnx.initializers.constant(0.0)

        # 100 nanometers / step_size nanometer
        self.final_size = int(100 / step_size)
        
        self.layer_sizes = [input_size] + list(hidden_sizes) + [self.final_size**2]
        
        self.layers = [
            nnx.Linear(i, o, kernel_init=dense_init, bias_init=bias_init, rngs=rngs) 
            for i, o in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
    
    
    @nnx.jit
    def __call__(self, x):
        x = jnp.reshape(x, (25,))

        for en, layer in enumerate(self.layers):
            x = layer(x)
            x = hardtanh(x)
        
        # Reshape the output
        x =  150.0 * jnp.reshape(x, (self.final_size, self.final_size))
        
        return x

