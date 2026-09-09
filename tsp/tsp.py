import numpy as np
import random
import matplotlib.pyplot as plt

POP_SIZE = 6       
ELITISM = 1         
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3
N_GENERATIONS = 5

cities = ['R', 'F', 'V', 'M', 'N']
n = len(cities)

# distance matrix 
dist_matrix = np.array([
    [0, 280, 530, 570, 225],
    [280, 0, 260, 310, 470],
    [530, 260, 0, 270, 690],
    [570, 310, 270, 0, 780],
    [225, 470, 690, 780, 0]])

# index to city -> R=0 , F=1, V=2, M=3, N=4
idx_to_city = {i: c for i, c in enumerate(cities)}

# calculate total distance of a tour
def tour_distance(tour_indices):  
    """Calculate total distance of a tour (list of indices)."""
    dist = 0
    for i in range(len(tour_indices)):
        a = tour_indices[i]
        b = tour_indices[(i + 1) % n] # back to the firt city
        dist += dist_matrix[a, b]
    return dist

def tour_to_names(tour_indices):
    return [idx_to_city[i] for i in tour_indices]

# generate initial random population func
def create_random_tour():
    tour = list(range(n))
    random.shuffle(tour)
    return tour

# initial random population
population = [create_random_tour() for _ in range(POP_SIZE)]

# selects random indexes (tournament size) then gets the one which has the best fitness value -> tsp  -> so the best value is the min distance
def tournament_select(pop, fitnesses):
    candidates = random.sample(range(len(pop)), TOURNAMENT_SIZE) # selects random indexes from the population 
    best_idx = min(candidates, key=lambda i: fitnesses[i]) # slect the best one with the best fitness function
    return pop[best_idx][:] # copy of the original population

# ox cross order: finds a senegment then fills that segement with one parnet and the rest is filled with the other parent with order
def ox(p1, p2):
    size = len(p1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2)) # selects two random indexes then sort them as sart and end (our segment)
    child[start:end] = p1[start:end] # copy parent no. one to the segment of the child
    pos = end % size # so it could loop over the parent no. two
    for gene in p2:                           # fill rest from parent2, preserving order
        if gene not in child: # if the number is not in child then copy, if i already is pass and go to next index
            child[pos] = gene
            pos = (pos + 1) % size
    return child

# picks two distinct random indexes of the child, then if the random number is less than the mutation rate, it swaps them
def swap_mutation(tour):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

# create the next generation
def next_generation(pop):
    fitnesses = [tour_distance(t) for t in pop] # find fitness value for each individual in the population
    sorted_idx = np.argsort(fitnesses) # sort by fitness value

    elites = [pop[i][:] for i in sorted_idx[:ELITISM]] # keep the best one
    new_pop = elites[:] # new population (has the best one in it)

    while len(new_pop) < POP_SIZE:
        # generational replacement with elitism
        # selct two parents using tournament selection
        p1 = tournament_select(pop, fitnesses)
        p2 = tournament_select(pop, fitnesses)
        # produce child with ox crossover
        child = ox(p1, p2)
        child = swap_mutation(child)
        new_pop.append(child)
    return new_pop

best_overall = None
best_dist = float('inf')
history = []

for gen in range(N_GENERATIONS):
    fitnesses = [tour_distance(t) for t in population]
    min_dist = min(fitnesses)
    best_tour = population[fitnesses.index(min_dist)][:]

    if min_dist < best_dist:
        best_dist = min_dist
        best_overall = best_tour[:]

    history.append(min_dist)

    print(f"GENERATION {gen+1}")
    for i, (tour, fit) in enumerate(zip(population, fitnesses), start=1):
        names = tour_to_names(tour)
        marker = "best" if fit == min_dist else ""
        print(f"  Ind {i:2d}: {names}  Distance = {fit:4d} km{marker}")
    print(f"Best: {min_dist} km")

    population = next_generation(population)

print(f"Final Best Solution: Tour: {tour_to_names(best_overall)} , Distance: {best_dist} km")

plt.plot(range(1, N_GENERATIONS+1), history)
plt.xlabel('generation')
plt.ylabel('best distance')
plt.grid(True)
plt.show()