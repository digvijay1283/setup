# Assignment 6: Travelling Salesman Problem using Ant Colony Optimization
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_cities = 10
cities = np.random.rand(n_cities, 2) * 100

# Distance matrix
dist = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        dist[i][j] = np.linalg.norm(cities[i] - cities[j])

# ACO parameters
n_ants, n_iter = 10, 50
alpha, beta = 1, 2        # importance of pheromone, importance of distance
rho, Q = 0.5, 100         # evaporation rate, pheromone constant
pheromone = np.ones((n_cities, n_cities))
eta = 1 / (dist + np.eye(n_cities))   # heuristic = 1/distance (eye avoids 1/0)

def tour_length(tour):
    return sum(dist[tour[i]][tour[(i + 1) % n_cities]] for i in range(n_cities))

best_tour, best_len = None, float('inf')
history = []

for it in range(n_iter):
    tours = []
    for ant in range(n_ants):
        tour = [np.random.randint(n_cities)]           # random start city
        while len(tour) < n_cities:
            cur = tour[-1]
            prob = (pheromone[cur] ** alpha) * (eta[cur] ** beta)
            prob[tour] = 0                             # visited cities not allowed
            prob = prob / prob.sum()
            tour.append(np.random.choice(n_cities, p=prob))
        tours.append(tour)

    # Evaporation
    pheromone *= (1 - rho)

    # Deposit: each ant adds Q / L on its edges
    for tour in tours:
        L = tour_length(tour)
        for i in range(n_cities):
            a, b = tour[i], tour[(i + 1) % n_cities]
            pheromone[a][b] += Q / L
            pheromone[b][a] += Q / L
        if L < best_len:
            best_len, best_tour = L, tour

    history.append(best_len)
    print(f"Iteration {it + 1:2d}: best length = {best_len:.2f}")

print("\nBest tour  :", best_tour + [best_tour[0]])
print(f"Best length: {best_len:.2f}")

route = best_tour + [best_tour[0]]
plt.subplot(1, 2, 1)
plt.plot(cities[route, 0], cities[route, 1], 'o-')
for i, (x, y) in enumerate(cities): plt.text(x, y, str(i))
plt.title('Best Tour')
plt.subplot(1, 2, 2)
plt.plot(history); plt.title('Convergence'); plt.xlabel('Iteration')
plt.tight_layout(); plt.show()
