
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mp
# from scipy import stats

# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error


# from math import sqrt

# df = pd.read_csv('FuelConsumptionCo2.csv')
# print(df.head())

# x = pd.read_csv('FuelConsumptionCo2.csv')
# y = pd.read_csv('FuelConsumptionCo2.csv')
# x = (df['ENGINESIZE'])
# y = (df['CO2EMISSIONS'])

# train_x = x[:80]
# train_y = y[:80]

# test_x = x[80:]
# test_y = y[80:]

# slope, intercept, r, p, std_err = stats.linregress(train_x, train_y)

# def myfunc(train_x):
#   return slope * train_x + intercept
# mymodel = list(map(myfunc, train_x))

# print(mymodel)
# print("\nINTERCEPT: ",intercept)
# print("\nSLOPE: ",slope)

# CO2 = myfunc(2.5)
# print("\nPREDICTION OF CO2 EMISSION AT 2.5",CO2)

# R2 = r2_score(test_x,test_y)
# ABSOLUTE = mean_absolute_error(test_x,test_y)
# Mean_Squared = mean_squared_error(test_x,test_y)

# print("\nR-SQUARED: ",R2)
# print("\nMEAN ABSOLUTE ERROR: ",ABSOLUTE)
# print("\nMEAN SQUARED ERROR: ",Mean_Squared)


import random
import math


def fitness(individual):
    x, y = individual
    error = abs(3*x + 2*y - 50)
    return 1 / (error + 1)

POPULATION_SIZE = 1000
GENE_LENGTH = 2
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.5
MAX_GENERATIONS = 100

population = []
for i in range(POPULATION_SIZE):
    individual = [random.randint(0, 100), random.randint(0, 100)]
    population.append(individual)

fitness_values = [fitness(individual) for individual in population]

for generation in range(MAX_GENERATIONS):
    parents = []
    for i in range(POPULATION_SIZE // 2):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        parents.append((parent1, parent2))

    offspring = []
    for parent1, parent2 in parents:
        if random.random() < CROSSOVER_RATE:
            crossover_point = random.randint(1, GENE_LENGTH - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    for individual in offspring:
        if random.random() < MUTATION_RATE:
            gene_to_mutate = random.randint(0, GENE_LENGTH - 1)
            individual[gene_to_mutate] = random.randint(0, 100)

    fitness_values = [fitness(individual) for individual in population]
    least_fit_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:len(offspring)]
    for i, individual in zip(least_fit_indices, offspring):
        population[i] = individual

    best_individual = max(population, key=fitness)
    print(f"Generation {generation}: {best_individual} with fitness {fitness(best_individual)}")

best_individual = max(population, key=fitness)
print(f"Final solution: {best_individual} with fitness {fitness(best_individual)}")