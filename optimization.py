import numpy as np
from models.mlp import mlp
from flax import nnx
from modules.params_utils import initialize_or_restore_params


# Model to be used for optimization
model_name = "MLP_baseline"
rngs = nnx.Rngs(42)
model = mlp(layer_sizes=[25, 32, 64, 128, 128, 256, 1], activation="relu", rngs=rngs)
model, checkpointer, ckpt_dir = initialize_or_restore_params(model, model_name, rank=0)

import random
import numpy as np
from deap import base, creator, tools

# Set Up the Genetic Algorithm
# Create the fitness and individual classes
creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=25)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    
    return np.abs(model(individual)-17.0),  # Must return a tuple

def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size - 1)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2

toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    random.seed(64)

    CXPB, MUTPB = 0.5, 0.2  # Crossover and mutation probabilities
    pop = toolbox.population(n=100)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evaluate the initial population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution loop
    for g in range(60):
        # Select and clone offspring
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population
        pop[:] = offspring

        # Record statistics
        record = stats.compile(pop)
        print(f"Gen {g}: {record}")

        # Update Hall of Fame
        hof.update(pop)

    return pop, stats, hof

# Run the algorithm
if __name__ == "__main__":
    print("Start Evolutionary Algorithm")
    pop, stats, hof = main()
    print("Hall of Fame:")
    print(hof[0])