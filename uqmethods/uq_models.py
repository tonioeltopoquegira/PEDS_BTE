from flax import nnx
import jax.numpy as jnp
from modules.training_utils import choose_activation
from uqmethods.pce import pcelowrank
import jax
import numpy as np


class logvarnet(nnx.Module):

    def __init__(self, hidden_sizes: list,  rngs: nnx.Rngs):
        super().__init__()

        layer_sizes = [25] + hidden_sizes + [1]
       
        dense_init = nnx.initializers.xavier_normal()
        bias_init = nnx.initializers.constant(0.0)
            
        self.layers = [
            nnx.Linear(i, o, kernel_init=dense_init, bias_init=bias_init, rngs=rngs)
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        self.activations = choose_activation("relu", len(self.layers))


    def __call__(self, x, training=False):
        for en, (layer, activation_fn) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            x = activation_fn(x) 
    
        return x


class lowrankpce(nnx.Module):

    def __init__(self, hidden_sizes:list, n_modes:int, pde_solver:callable, rngs: nnx.Rngs):
        super().__init__()

        self.pde = pde_solver

        self.rngs = rngs

        hidden_sizes_var = [25] + hidden_sizes + [25]
        hidden_sizes_mode = [25] + hidden_sizes + [n_modes * 25]
       
        # Variance 
        dense_init = nnx.initializers.xavier_normal()
        bias_init = nnx.initializers.constant(0.0)
        self.layers_var = [
            nnx.Linear(i, o, kernel_init=dense_init, bias_init=bias_init, rngs=rngs)
            for i, o in zip(hidden_sizes_var[:-1], hidden_sizes_var[1:])
        ]
        self.act_var = choose_activation("relu", len(self.layers_var))

        # Other modes

        self.layers_mode = [
            nnx.Linear(i, o, kernel_init=dense_init, bias_init=bias_init, rngs=rngs)
            for i, o in zip(hidden_sizes_mode[:-1], hidden_sizes_mode[1:])
        ]
        self.act_mode = choose_activation("relu", len(self.layers_mode))
        

    def __call__(self, x, mean, training=False, key=42):
        logvar = x
        u = x

        for en, (layer, activation_fn) in enumerate(zip(self.layers_mode, self.act_mode)):
            u = layer(u)
            u = activation_fn(u)
        
        for en, (layer, activation_fn) in enumerate(zip(self.layers_var, self.act_var)):
            logvar = layer(logvar)
            logvar = activation_fn(logvar)  # Apply the activation function for this layer

        var = jnp.exp(logvar)

        #print("var",var[0])
        #print("u",u[0])

        # Perform Polynomial Chaos Expansion - Not full now
        k_mean, k_var = pcelowrank(mean, var, u, self.pde, 10, self.rngs) 

        return k_mean, k_var
    

def uqamethod(mean: jnp.ndarray,
              var:  jnp.ndarray,
              u:    jnp.ndarray,
              pdesolver: callable) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute mean and variance of k = pdesolver( grid(mean + noise) )
    under x ~ N(mean, diag(var) + u u^T), assuming pdesolver->integrator is linear in x.

    Args:
      mean: shape (25,) vector μ
      var:  shape (25,) vector of variances σ_i^2
      u:    shape (25,) vector for the rank-1 part
      pdesolver: function f(grid: jnp.ndarray[5,5]) -> scalar k

    Returns:
      k_mean: scalar E[k]
      k_var:  scalar Var[k]
    """
    # 1) wrap solver to go from flat x -> k
    def k_of_x(x: jnp.ndarray) -> jnp.ndarray:
        grid = x.reshape((5,5))
        return pdesolver(grid)

    # 2) compute the mean by one forward solve
    k_mean = k_of_x(mean)

    # 3) compute the gradient ∇_x k at x=mean (one backward pass)
    grad_k = jax.grad(k_of_x)(mean)   # shape (25,)

    # 4) propagate variance exactly for a linear map:
    #    Var[k] = grad_k^T ⋅ Σ ⋅ grad_k
    #           = sum_i var[i] * grad_k[i]^2   (diagonal)
    #             + (u^T grad_k)^2             (rank-1 piece)
    k_var = jnp.dot(var, grad_k**2) + (jnp.dot(u, grad_k))**2

    return k_mean, k_var