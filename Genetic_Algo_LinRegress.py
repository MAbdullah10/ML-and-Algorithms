'''
AI THEORY ASSIGNMENT 3
USE GENETIC ALGO TO OPTIMIZE LINEAR REGRESSION

GROUP MEMBERS
    FA21-BCS-030    MOIZ AHMAD
    FA21-BCS-048    AZEEM AZAM
    FA21-BCS-135    M.ABDULLAH
'''
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GA:
    def __init__(self, individualSize, populationSize):
        self.population = dict() #Dictionary to store individuals and their fitness values.
        self.individualSize = individualSize #Number of coefficients.
        self.populationSize = populationSize #Number of individuals in the population.
        self.totalFitness = 0 
        #Loop to create the initial population with random coefficients between -1 and 1 and a fitness value of 0.
        i = 0
        while i < populationSize:
            coefficients = [random.uniform(-1, 1) for _ in range(individualSize)]
            self.population[i] = [coefficients, 0]
            i += 1

    def updatePopulationFitness(self): #Calculates and updates the fitness of each individual in the population.
        self.totalFitness = 0
        #Loops through each individual, computes its fitness using the linear_regression_fitness function, updates the individual's fitness, and adds it to the total fitness.
        for individual in self.population:
            fitness = linear_regression_fitness(X_train_scaled, y_train, self.population[individual][0])
            self.population[individual][1] = fitness
            self.totalFitness += fitness

    def selectParents(self): #Implements roulette wheel selection to choose parents based on fitness.
        rouletteWheel = []
        min_fitness = min(self.population.values(), key=lambda x: x[1])[1] #Minimum fitness in the population to handle negative values.
        total_h_n = sum(max(0, x[1] - min_fitness + 1) for x in self.population.values()) #Sum of adjusted fitness values to normalize the selection probabilities.
        if total_h_n == 0:
            return  
        #For each individual, calculates its normalized fitness and appends it to rouletteWheel multiple times proportional to its fitness.
        for individual, data in self.population.items():
            fitness = max(0, data[1] - min_fitness + 1)
            individualLength = round(self.populationSize * fitness / total_h_n)
            rouletteWheel.extend([individual] * individualLength)
        #Selects parents based on rouletteWheel and creates a new generation.
        parentIndices = random.choices(rouletteWheel, k=self.populationSize) 
        newGeneration = {i: self.population[index].copy() for i, index in enumerate(parentIndices)}
        self.population = newGeneration.copy()
        self.updatePopulationFitness()

    def generateChildren(self, crossOverProbability): #Performs crossover to generate children from selected parents with a given probability.
        for i in range(0, self.populationSize, 2): #Loops through the population in pairs.
            #If a random number is less than crossOverProbability and there is a valid pair, performs crossover.
            if random.random() < crossOverProbability and (i + 1) < self.populationSize: 
                parent1, parent2 = self.population[i][0], self.population[i + 1][0]
                child1, child2 = [], []
                #For each gene in the parents, randomly decides whether to swap the genes between the children.
                for gene1, gene2 in zip(parent1, parent2):
                    if random.random() < 0.5:
                        child1.append(gene1)
                        child2.append(gene2)
                    else:
                        child1.append(gene2)
                        child2.append(gene1)
                self.population[i] = [child1, 0] #Updates the population with the new children and resets their fitness to 0.
                self.population[i + 1] = [child2, 0]
        self.updatePopulationFitness()

    def mutateChildren(self, mutationProbability): #Introduces mutations in the population with a given probability.
        for individual in self.population:
            coefficients = self.population[individual][0]
            for i in range(len(coefficients)):
                #For each coefficient, if a random number is less than mutationProbability, modifies the coefficient slightly by adding a random value between -0.1 and 0.1.
                if random.random() < mutationProbability:
                    coefficients[i] += random.uniform(-0.1, 0.1)
            self.population[individual][0] = coefficients
        self.updatePopulationFitness()

def linear_regression_fitness(X_train, y_train, coefficients): #Computes the fitness of an individual based on the mean squared error of the predictions
    y_pred = np.dot(X_train, coefficients[1:]) + coefficients[0] #X_train is multiplied by the coefficients (excluding the bias term), and the bias term is added to obtain predictions.
    return mean_squared_error(y_train, y_pred)

data = pd.read_csv("./Linear Regression - Sheet1.csv")
data.dropna(inplace=True) #Drops any rows with missing values.
X = data[['X']].values
y = data['Y'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Splits the data into training and testing sets with 80% training and 20% testing.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

individualSize, populationSize = X_train_scaled.shape[1] + 1, 10 #Determines the size of each individual (number of features + 1 for the bias term) and sets the population size.
instance = GA(individualSize, populationSize) 

best_fitness_history = [] #Initializes variables to track the best fitness history and convergence criteria.
#Runs the genetic algorithm for up to 100 iterations.
convergence_threshold = 1e-6 
convergence_iterations = 10
prev_best_fitness = float('inf')
convergence_count = 0
for i in range(100): 
    instance.selectParents()
    instance.generateChildren(0.8)
    instance.mutateChildren(0.1)
    best_solution = min(instance.population.values(), key=lambda x: x[1]) #Identifies the best solution in the current population.
    best_fitness = best_solution[1]
    best_fitness_history.append(best_fitness)
    print("Iteration:", i)
    print("Best solution (coefficients):", best_solution)
    print("Best fitness (RMSE):", best_fitness)
    print("Best fitness history:", best_fitness_history)
    #Checks for convergence based on the change in best fitness and terminates if convergence is reached.
    if abs(prev_best_fitness - best_fitness) < convergence_threshold:
        convergence_count += 1
    else:
        convergence_count = 0
    if convergence_count >= convergence_iterations:
        print("Convergence reached. Terminating.")
        break
    prev_best_fitness = best_fitness

plt.plot(best_fitness_history, marker='o')
plt.title('Best Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (RMSE)')
plt.grid(True)
plt.show()
