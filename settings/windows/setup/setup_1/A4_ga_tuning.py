# Assignment 4: Optimize ML model parameters using a Genetic Algorithm
import random
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

random.seed(42)
X, y = load_breast_cancer(return_X_y=True)

# Chromosome = [n_estimators, max_depth]
def random_chromosome():
    return [random.randint(10, 100), random.randint(1, 20)]

def fitness(ch):  # cross-validation accuracy
    model = RandomForestClassifier(n_estimators=ch[0], max_depth=ch[1], random_state=42)
    return cross_val_score(model, X, y, cv=3).mean()

def select(pop, scores):  # tournament selection (size 3)
    picks = random.sample(range(len(pop)), 3)
    best = max(picks, key=lambda i: scores[i])
    return pop[best][:]

def crossover(p1, p2):  # single-point crossover
    return [p1[0], p2[1]], [p2[0], p1[1]]

def mutate(ch, rate=0.2):  # random reset mutation
    if random.random() < rate: ch[0] = random.randint(10, 100)
    if random.random() < rate: ch[1] = random.randint(1, 20)
    return ch

POP_SIZE, GENERATIONS = 8, 5
population = [random_chromosome() for _ in range(POP_SIZE)]

for gen in range(1, GENERATIONS + 1):
    scores = [fitness(ch) for ch in population]
    best_i = scores.index(max(scores))
    best = population[best_i][:]
    print(f"Generation {gen}: best = {best}  accuracy = {scores[best_i]:.4f}")

    new_pop = [best]                          # elitism: keep the best
    while len(new_pop) < POP_SIZE:
        c1, c2 = crossover(select(population, scores), select(population, scores))
        new_pop += [mutate(c1), mutate(c2)]
    population = new_pop[:POP_SIZE]

print(f"\nBest parameters: n_estimators = {best[0]}, max_depth = {best[1]}")
print(f"Best accuracy   : {scores[best_i]:.4f}")
