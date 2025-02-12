import os
import numpy as np
from models.mlp import mlp
from models.peds import PEDS
from flax import nnx
from modules.params_utils import initialize_or_restore_params


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


def main(model, target):

    
    def evaluate(individual):
        if isinstance(model, PEDS):

            individual = individual.reshape((1,5,5))
            kappa, _ = model(individual)
        
        else:
            kappa = model(individual)

        return np.abs(kappa-target),  # Must return a tuple

    toolbox.register("evaluate", evaluate)

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
    for g in range(40):
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

    import pandas as pd
    import os
   
    
    model_name = "PEDS_gauss"
    results_file = f"data/optimization/{model_name}/evolutionary_geometries.csv"
    os.makedirs(f"data/optimization/{model_name}", exist_ok=True)

    # Load existing results if the file exists, otherwise create a new DataFrame
    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
    else:
        results = pd.DataFrame(columns=["kappa_target", "geometries"])

    # Initialize the model
    model = PEDS(resolution = 20, learn_residual= False, hidden_sizes= [32, 64, 128], activation="relu", solver="gauss") # parameters: 60k
    model, checkpointer = initialize_or_restore_params(model, model_name, rank=0)

    # Define the target kappas
    kappas_target = [0.0, 13.0, 17.0, 20.0, 23.0, 26.0, 33.0, 37.0, 45.0, 70.0, 100.0, 160.0]

    # Iterate through each kappa target
    for kappa in kappas_target:
        # Skip if kappa is already in the results
        if kappa in results["kappa_target"].values:
            print(f"Skipping kappa {kappa}, already processed.")
            continue

        print(f"Start Evolutionary Algorithm for {kappa}")
        pop, stats, hof = main(model, kappa)

        print("Hall of Fame:")
        hof_array = np.array(hof[0])
        print(hof_array)

        # Append the results to the DataFrame
        results = results._append({"kappa_target": kappa, "geometries": hof_array.tolist()}, ignore_index=True)

        # Save the DataFrame to the CSV file after every iteration
        results.to_csv(results_file, index=False)

    print("Optimization completed.")
