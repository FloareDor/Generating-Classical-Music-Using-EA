import numpy as np
import matplotlib.pyplot as plt

def eggholder(x):
    """The Eggholder function to optimize"""
    return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0]/2 + (x[1]+47)))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1]+47))))

def differential_evolution(func, bounds, pop_size=100, max_gen=1000, mutation_factor=0.5, crossover_prob=0.7):
    """Implementation of the differential evolution optimization algorithm"""
    # Initialize the population of candidate solutions
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, len(bounds[0])))
    # Evaluate the initial population
    fitness = np.apply_along_axis(func, 1, population)
    best_solutions = []
    for i in range(max_gen):
        for j in range(pop_size):
            # Select three different individuals randomly
            idx = np.random.choice(np.arange(pop_size), 3, replace=False)
            x1, x2, x3 = population[idx]
            # Mutate the solution
            mutant = x2 + mutation_factor * (x3 - x1)
            # Crossover with the current individual with a probability of crossover_prob
            cross_points = np.random.rand(len(bounds[0])) < crossover_prob
            trial = np.where(cross_points, mutant, population[j])
            trial_fit = func(trial)
            # Replace the current individual with the trial if the trial is better
            if trial_fit < fitness[j]:
                population[j] = trial
                fitness[j] = trial_fit
        # Store the best solution found in this generation
        best_idx = np.argmin(fitness)
        best_solutions.append(population[best_idx])
    # Return the best solutions found in all generations
    return best_solutions

bounds = (np.array([-512, -512]), np.array([512, 512]))
best_solutions = differential_evolution(eggholder, bounds)
best_fitness = [eggholder(solution) for solution in best_solutions]

# Plot the best solution found in each generation
plt.plot(best_fitness)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()

# Print the best solution found
best_idx = np.argmin(best_fitness)
best_solution = best_solutions[best_idx]
print("Best solution:", best_solution)
print("Best fitness:", best_fitness[best_idx])
