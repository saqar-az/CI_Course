import numpy as np
import random
import operator
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms

np.random.seed(42)
x1 = np.random.uniform(-5, 5, 300)
x2 = np.random.uniform(-5, 5,300)
# the true function
y_true = np.where(x1 > 0, x1 * np.sin(x2), x2 * np.cos(x1))
# adding random noise to y
noise = np.random.normal(0, 0.01, 300)
y = y_true + noise
# the dataset
dataset = list(zip(x1, x2, y))

# grammer: each node has two children that are float
pset = gp.PrimitiveSetTyped("MAIN", [float, float], float)
pset.renameArguments(ARG0='x1', ARG1='x2')
# grammer: each of add and mlutiply operaters gets 2 childern that are float and gives out a fliat
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.mul,  [float, float], float)

# check for x1>0
def is_positive(x):
    return x > 0.0
pset.addPrimitive(is_positive, [float], bool) 

def if_then_else(cond, a, b):
    return a if cond else b
# takes 3 inputs that are bool, foat and float and returns a float
pset.addPrimitive(if_then_else, [bool, float, float], float)

# sin and cos functions have 1 child
pset.addPrimitive(np.sin, [float], float)
pset.addPrimitive(np.cos, [float], float)
# terminla nodes
pset.addTerminal(0.0, float)
pset.addTerminal(1.0, float)
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)      

# get fitness value for a tree
def evaluate(tree):
    func = gp.compile(tree, pset)
    mse = 0.0
    for x1_val, x2_val, y_val in dataset:
        try:
            pred = func(x1_val, x2_val)
            if np.isnan(pred) or np.isinf(pred):
                pred = 1e6
        except:
            pred = 1e6
        mse += (pred - y_val) ** 2
    mse /= len(dataset)
    size = len(tree)
    fitness = - (mse + 0.01 * size)
    return (fitness,)

# ramped half-and-half
def ramped_half_and_half(pset, min_depth, max_depth):
    depth = random.randint(min_depth, max_depth)
    return gp.genHalfAndHalf(pset, min_=depth, max_=depth)

# create the fitness and the individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
# register the generation functions into a toolbox.
toolbox = base.Toolbox()
# ramped half-and-half: depth from min to max
toolbox.register("expr", ramped_half_and_half, pset=pset, min_depth=2, max_depth=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# evaluation
toolbox.register("evaluate", evaluate)
# tournamnet selection
toolbox.register("select", tools.selTournament, tournsize=3)
# one point co: selects a node the swaps the subtree
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)

def combined_mutation(ind):
    if random.random() < 0.5:
        # selects a node then replace it's subtree with a rendom subtree
        ind, = gp.mutUniform(ind, expr=toolbox.expr_mut, pset=pset)
    else:
        # selects a node then replace it with a terminal node
        ind, = gp.mutShrink(ind)
    return ind,

toolbox.register("mutate", combined_mutation)

# check the height of the trees after co/ mutation
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

# generating population
pop = toolbox.population(n=500)
# kepping the best trees with the highest fitness function
hof = tools.HallOfFame(1)
# getting the fitness value for each tree and then getting the max and avg values
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

# genetring: with co probability, mutation probability and generation number
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.5, ngen=200,
                               stats=stats, halloffame=hof, verbose=True)

# print each generation
gen = log.select("gen")
avg_fitness = log.select("avg")
max_fitness = log.select("max")
plt.figure(figsize=(10, 5))
plt.plot(gen, avg_fitness, label="Average Fitness")
plt.plot(gen, max_fitness, label="Best Fitness")
plt.xlabel("generation")
plt.ylabel("fitness")
plt.legend()
plt.grid(True)
plt.show()

# best tree
best = hof[0]
print("best tree:", best)

def tree_to_stirng(tree):
    # terminal node (x1, x2 or const)
    if len(tree) == 1: 
        return str(tree[0])
    # internal node 
    op = tree[0]
    if op == 'add':
        return f"({tree_to_stirng(tree[1])} + {tree_to_stirng(tree[2])})"
    elif op == 'sub':
        return f"({tree_to_stirng(tree[1])} - {tree_to_stirng(tree[2])})"
    elif op == 'mul':
        return f"({tree_to_stirng(tree[1])} * {tree_to_stirng(tree[2])})"
    elif op == 'sin':
        return f"sin({tree_to_stirng(tree[1])})"
    elif op == 'cos':
        return f"cos({tree_to_stirng(tree[1])})"
    elif op == 'if_then_else':
        return f"if_then_else({tree_to_stirng(tree[1])}, {tree_to_stirng(tree[2])}, {tree_to_stirng(tree[3])})"
    elif op == 'gt':
        return f"({tree_to_stirng(tree[1])} > {tree_to_stirng(tree[2])})"
    else:
        return str(tree)  

# tree to string
print(tree_to_stirng(best))
print("Best fitness:", best.fitness.values[0])    