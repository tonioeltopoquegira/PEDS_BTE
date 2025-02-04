from flax import nnx
import jax.numpy as jnp
from models.mlp import mlp
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid
from test_code import optimized_conductivity_grid_jax



# An easy PEDS Wrapper
class PEDS(nnx.Module):

    def __init__(self, resolution:int, 
                learn_residual: bool, hidden_sizes:list, activation:str,
                solver:str):
        super().__init__()

        # 100 nanometers / step_size nanometer
        self.resolution = resolution
        self.layer_sizes = [25] + hidden_sizes + [resolution**2]
        self.activation = activation
        self.learn_residual = learn_residual

        # Create model
        key = nnx.Rngs(42)

        self.generator = mlp(layer_sizes=self.layer_sizes, activation = activation, rngs=key, last_activation=True) # 
        
        # Low Fidelity Solver
        self.lowfidsolver = lowfid(solver=solver, iterations=1000)
    
    def __call__(self, pores): # Here

        batch_size = pores.shape[0]

        pores = jnp.reshape(pores, (batch_size, 25)) 

        # Process data through the generator (MLP)
        conductivity_generated = nnx.jit(self.generator)(pores)

        # Reshape the output
        conductivity_generated = jnp.reshape(conductivity_generated, (batch_size, self.resolution, self.resolution))

        # Rescale and Adjust
        if self.activation == "hardtanh":
            conductivity_generated *= 150.0

        if self.learn_residual:
            conductivities = optimized_conductivity_grid_jax(pores, self.resolution)
            conductivity_final = conductivity_generated+conductivities 
        else:
            conductivity_final = conductivity_generated

        conductivity_final = jnp.maximum(conductivity_final, 1e-12)
        kappa = self.lowfidsolver(conductivity_final) 

        
        return kappa, conductivity_generated

