import numpy as np
import jax
import jax.numpy as jnp
import optax
from sqlitedict import SqliteDict
import pickle

from matinverse import Geometry2D,BoundaryConditions, BTE
from matinverse.material import RTA2D

def bte_solver(conductivity):

    # Define geometry
    L = 1e-6
    size = [L, L]
    N = conductivity.shape[1]
    cond = conductivity.reshape((conductivity.shape[0], N**2))


    grid = [N, N]

    geo = Geometry2D(grid, size, periodic=[True, True])  

    print(geo.N)
    mat = RTA2D(filename='OpenBTE/openbte/materials/Si_rta')
    bte = BTE(geo, mat=mat)

    bcs = BoundaryConditions(geo)
    bcs.periodic('y', lambda batch, space, t: 1.0)
    bcs.periodic('x', lambda batch, space, t: 0.0)

   
    rho = jnp.full((geo.N,), 1.0)   # a length‐10000 array
    output = bte(rho, bcs)
    #output = bte(cond, bcs, batch_size= cond.shape[0]) # HERE ERROR

    T = output['T']

    T = T.reshape((conductivity.shape))

    # Extract relevant quantities
    kappa_effective = output['kappa']


    return T, kappa_effective

conductivity = jnp.ones((1, 10, 10))

from sqlitedict import SqliteDict
import pickle

'''db_path = "OpenBTE/openbte/materials/Si_rta.db"

# Open the SQLite database in read-only mode
with SqliteDict(db_path, flag='r') as db:
    print("Keys in DB:", list(db.keys()))

    # Example: print the shape and some stats for each key
    for key in db:
        value = db[key]
        print(f"{key}: type={type(value)}")

        # Try printing shape/summary if it's an array
        if hasattr(value, 'shape'):
            print(f"  shape={value.shape}, mean={value.mean()}, max={value.max()}")
        elif isinstance(value, list):
            print(f"  len={len(value)}")'''



res = bte_solver(conductivity)

print(res)

#res = bte_solver(conductivity)


'''def loss_and_aux(p):
    #density = convert_pores_jax(p, grid_size=100)
    #print(density.shape)
    k = highfidelity_solver(p, step_size=2)

    loss = (k - 10.0)**2
    return loss

# 4) set up Adam
learning_rate = 1e-2
optimizer = optax.adam(learning_rate)

# initial guess: e.g. 5×5 half‐filled
p0 = jnp.full((25,), 0.5)

opt_state = optimizer.init(p0)

# 5) one optimization step
def step(p, opt_state):
    loss, grad = jax.value_and_grad(loss_and_aux)(p)
    updates, opt_state = optimizer.update(grad, opt_state, p)
    p = optax.apply_updates(p, updates)
    return p, opt_state, loss, k

# 6) run the loop
p, opt_state = p0, opt_state
for i in range(200):
    p, opt_state, loss, k = step(p, opt_state)
    if i % 20 == 0:
        print(f"iter {i:3d}: loss={loss:.4f}, k={k}")

# 7) final
print("Optimized pore params:", p'''