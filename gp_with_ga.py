#!/usr/bin/env python3
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")

# -------------------------
# 1) LOAD DATA (same as before)
# -------------------------
data = np.load("data/highfidelity/high_fidelity_2_20000.npz", allow_pickle=True)
X = data["pores"].astype(np.float32)    # shape (N,25)
y = data["kappas"].astype(np.float32)   # shape (N,)

rng = np.random.default_rng(46)
perm = rng.permutation(len(X))
test_idx = perm[:1000]
train_idx = perm[1000:2000]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"Training on {len(X_train)} points, testing on {len(X_test)} points")

# -------------------------
# 2) FIT GP
# -------------------------
kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-6)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
gp.fit(X_train, y_train)
print("GP fitted. Learned kernel:", gp.kernel_)

# -------------------------
# 3) DEFINE GENETIC ALGORITHM
# -------------------------
def find_pattern_ga(gp, target, n_pop=175, n_gen=50, cx_pb=0.5, mut_pb=0.2):
    """
    Use GA to find a binary pattern (length 25) whose GP-predicted mean
    is closest to target.
    """
    import random
    # 1) Define fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Binary genome of length 25
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 25)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness function: squared difference between GP prediction and target
    def eval_pattern(individual):
        x = np.array(individual).reshape(1, -1)
        pred = gp.predict(x)[0]
        return ((pred - target) ** 2,)

    toolbox.register("evaluate", eval_pattern)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)  # keep best individual

    # Run GA
    algorithms.eaSimple(pop, toolbox, cxpb=cx_pb, mutpb=mut_pb, ngen=n_gen,
                        halloffame=hof, verbose=False)

    best_ind = np.array(hof[0])
    best_loss = eval_pattern(hof[0])[0]
    return best_ind, best_loss

# -------------------------
# 4) DEMO INVERSION
# -------------------------
if __name__ == "__main__":
    for T in [59.99]:
        pattern25, loss = find_pattern_ga(gp, target=T, n_pop=175, n_gen=60)
        print(f"\nTarget kappa = {T}")
        print(" Best 1D pattern:", pattern25.tolist())
        print(f" Surrogate-predicted squared error: {loss:.6f}")


# 12.01, [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], 0.000014, 12.87
# 15.0 , [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], 0.00 15.04
# 20.0 [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0] 0.00 19.69
# 30.0 [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0], 0.00124 31.077
# 44.98 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.000191 42.889
# 59.99 [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.000069 60.63

# change seed and changed number

# 12.01 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] 0.000085 13.95881375699333
# 15.00 [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1] 0.000000 14.754908782972416
# 20.00 [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1] 0.0000 19.548119000549796
# 30.00 [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0] 0.00001 30.22495077526478
# 44.98 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0] 0.00014 41.67892715170844
# 59.99 [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 0.000100 57.46733215645544