import numpy as np
import random
import matplotlib.pyplot as plt

def eggholder(X):
	# X is a np.array
	return (-(X[1] + 47) * np.sin(np.sqrt(abs(X[0]/2 + (X[1] + 47)))) -X[0] * np.sin(np.sqrt(abs(X[0] - (X[1] + 47)))))


def DifferentialEvolution(populationSize : int, generations : int):
	# CONSTANTS as defined by the question
	dimensionSize = 2 # (x, y)
	bounds = [(-512, 512), (-512, 512)]
	crossoverProbability = 0.8
	K = 0.5

	generations_AvgFitness = []
	generations_GlobMinFitness = []

	# Initialize random parents
	parents = [ np.array([random.uniform(bounds[j][0], bounds[j][1]) for j in range(dimensionSize)]) for i in range(populationSize)]

	generationNumber = 0

	while (generationNumber < generations):
		generationNumber += 1
		children = [] # The new children will be added here
		F = random.uniform(-2.0, 2.0) # Our F is to be randomly generated every generation

		for index, vector in enumerate(parents):
			# Remove the parent vector so that R1, R2 and R3 won't be selected as the parent vector
			pruned_parents = parents.copy()
			pruned_parents.pop(index)

			# This while loop exists only if the Vector_Trial is out of bounds (i.e. not between (-512, 512))
			while (True):
				Vector_R1, Vector_R2, Vector_R3 = random.sample(pruned_parents, 3)

				#  Mutant Vector
				Vector_Mutant = vector + K * (Vector_R1 - vector) + F * (Vector_R2 - Vector_R3)

				# Trial Vector
				Vector_Trial = np.array([0.0 for i in range(dimensionSize)])

				# Crossover
				for gene in range(dimensionSize):
					crossoverRealtime = random.random()

					if crossoverRealtime < crossoverProbability:
						Vector_Trial[gene] = Vector_Mutant[gene]
					else:
						Vector_Trial[gene] = vector[gene]

				# Check if the Trial Vector is in bounds (i.e. between (-512, 512))
				flagInBounds = True
				for i in range(dimensionSize):
					if not ((bounds[i][0] < Vector_Trial[i]) and (Vector_Trial[i] < bounds[i][1])):
						flagInBounds = False
						break

				# Elitism: Get the better vector w.r.t. fitness
				if flagInBounds:
					if eggholder(Vector_Trial) < eggholder(vector):
						children.append(Vector_Trial)
					else:
						children.append(vector)
					break

		# Calculate values for plotting
		parents_values = [ (eggholder(i), i) for i in parents ]
		parents_values.sort()

		generations_GlobMinFitness.append(parents_values[0][0])
		
		average = 0
		for child in parents_values:
			average += child[0]

		generations_AvgFitness.append(average/populationSize)

		parents = children.copy()

	plt.cla()
	plt.plot(generations_GlobMinFitness, color='green', linestyle='-', label="Global Minimum")
	plt.plot(generations_AvgFitness, color='blue', linestyle='-', label="Average")
	plt.title("Population: " + str(populationSize) + " | Generations : " + str(generations) + " | Global Minimum: " + str(generations_GlobMinFitness[len(generations_GlobMinFitness)-1]),
		fontdict={'fontsize' : 8})
	plt.legend()
	plt.locator_params(axis='y', nbins=10)
	plt.xlabel("Generation")
	plt.ylabel("Fitness")
	plt.savefig("DiffEvo-Pop_"+str(populationSize)+"_Gens_"+str(generations)+".png", dpi=500)


if __name__ == '__main__':
	print ("Differential Evolution - Eggholder Function")

	populationSize = [20, 50, 100, 200] 
	generations = [50, 100, 200]

	for generation in generations:
		for population in populationSize:
			DifferentialEvolution(population, generation)