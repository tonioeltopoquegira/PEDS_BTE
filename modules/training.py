import time
import jax
import jax.numpy as jnp
import optax
from flax import nnx


from jax.sharding import Mesh, PartitionSpec, NamedSharding

from modules.params_utils import save_params
from modules.training_utils import data_loader, print_generated, clip_gradients





def train_model(dataset, epochs, generator, lowfidsolver, checkpointer, ckpt_dir): 

    n_devices = jax.device_count()

    # Parameters Training
    batch_size = 100

    # Optimizer
    # SGD + 1e-3 very good with iterative

    optimizer = nnx.Optimizer(generator, optax.sgd(1e-5)) # large parameters tensor... takes a while with ADAM
    #schedule = optax.schedules.cosine_decay_schedule(init_value = , decay_steps=)
    step_size = 1

    N = int(100 / step_size)
    #size_square = int(10 * 1 / step_size)
    #half_size_square = size_square // 2
    #subgrid = jnp.ones((size_square, size_square)) * 1e-9
    #indices = jnp.stack(jnp.meshgrid(jnp.arange(5), jnp.arange(5)), axis=-1).reshape(-1, 2)

    
    def train_step(pores, conductivities, kappas, batch_n, epoch): # sharded pores and kappas
        
        def loss_fn(generator):
            
            #mlp_time = time.time()
            conductivity_res = nnx.vmap(generator)(pores)
            #print("Residual generation time", time.time()-mlp_time)
            
            #t_solv = time.time()
            kappa = lowfidsolver(conductivity_res+conductivities) 
            #print("Solver:", time.time()-t_solv)

            #print(f"Predicted: {kappa[:2]}, Target: {kappas[:2]}")

            residuals = (kappa - kappas) 

            if batch_n == 0:
                print_generated(conductivities, conductivity_res, epoch)
           
            return jnp.mean(residuals** 2)
        
        #t_forward_back = time.time()

        loss, grads = nnx.value_and_grad(loss_fn)(generator)

        
        
        #print(f"Forward+Backward: {time.time()-t_forward_back}")
        
        return loss, grads

    print("Training...")
    for epoch in range(epochs):
        grads = None
        total_loss = 0.0  # Initialize total loss for the epoch
        
        batch_size = batch_size // n_devices

        for en, batch in enumerate(data_loader(*dataset, batch_size=batch_size)):
            
            pores, conductivities, kappas = batch

            t_batch = time.time()
            loss, grads_new = train_step(pores, conductivities, kappas, en, epoch)
            
            total_loss += loss  # Add loss for the current batch
            
            if grads is None:
                grads = grads_new
            else:
                grads = jax.tree_util.tree_map(lambda g, gn: g + gn, grads, grads_new)
            
            grads = clip_gradients(grads, clip_value=10.0)
            #print("Time for batch:", time.time()-t_batch)

        # Print the average loss at the end of each epoch
        print(f"Grads, {grads}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / batch_size}")

        optimizer.update(grads)
    
    save_params(generator, checkpointer)
    
    

    
    



