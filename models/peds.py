from flax import nnx
import jax.numpy as jnp
import jax
from models.mlp import mlp
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid
from solvers.low_fidelity_solvers.base_conductivity_grid_converter import conductivity_original_wrapper


# An easy PEDS Wrapper
class PEDS(nnx.Module):

    def __init__(self, resolution:int, 
                adapt_weights: bool, learn_residual: bool, hidden_sizes:list, activation:str,
                solver:str, initialization:str, seed:int = 42):
        super().__init__()

        # 100 nanometers / step_size nanometer
        self.adapt_weights = adapt_weights
        self.resolution = resolution
        self.layer_sizes = [25] + hidden_sizes + [resolution**2]
        self.activation = activation
        self.learn_residual = learn_residual
       
        rng = jax.random.PRNGKey(seed)
        key = nnx.Rngs({'params': rng})

        if self.adapt_weights:
            self.wnn = mlp(layer_sizes=[25, 1], activation="sigmoid", rngs=key, initialization="he")
        self.generator = mlp(layer_sizes=self.layer_sizes, activation = activation, rngs=key, initialization=initialization) # 
        
        # Low Fidelity Solver
        self.lowfidsolver = lowfid(solver=solver, iterations=1000)
    
    def __call__(self, pores, training=False):
        batch_size = pores.shape[0]
        pores = jnp.reshape(pores, (batch_size, 25))

        # Run the generator
        conductivity_generated_raw = nnx.jit(self.generator, static_argnames=("training",))(pores, training)
        conductivity_generated_raw = jnp.reshape(conductivity_generated_raw, (batch_size, self.resolution, self.resolution))

        conductivity_generated = conductivity_generated_raw  # default to raw unless residual is applied

        if self.learn_residual:
            conductivities = conductivity_original_wrapper(pores, self.resolution)
            if self.adapt_weights:
                w = nnx.jit(self.wnn, static_argnames=("training",))(pores, training)
                w1 = jnp.expand_dims(w, -1)
                conductivity_generated_raw = w1 * conductivity_generated_raw
                
                w2 = 1 - w1
                conductivities = w2 * conductivities
            
            conductivity_generated = conductivities + conductivity_generated_raw

        if self.activation == "relu":
            conductivity_generated = jnp.maximum(conductivity_generated, 1e-16)

        kappa = self.lowfidsolver(conductivity_generated)
        
        return kappa, conductivity_generated_raw  # <- return only what the NN generated
