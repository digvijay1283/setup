# Assignment 7: Data clustering using Particle Swarm Optimization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
K, n_particles, n_iter = 3, 20, 100
w, c1, c2 = 0.7, 1.5, 1.5           # inertia, cognitive, social

def sse(centroids):  # fitness = sum of squared distances to nearest centroid
    d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.sum(np.min(d, axis=1) ** 2)

# Each particle = K centroids (shape K x 2)
lo, hi = X.min(axis=0), X.max(axis=0)
pos = np.random.uniform(lo, hi, size=(n_particles, K, 2))
vel = np.zeros_like(pos)
pbest = pos.copy()
pbest_score = np.array([sse(p) for p in pos])
gbest = pbest[np.argmin(pbest_score)].copy()
gbest_score = pbest_score.min()

for it in range(n_iter):
    for i in range(n_particles):
        r1, r2 = np.random.rand(K, 2), np.random.rand(K, 2)
        # Velocity and position update
        vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gbest - pos[i])
        pos[i] = pos[i] + vel[i]

        score = sse(pos[i])
        if score < pbest_score[i]:              # update personal best
            pbest[i], pbest_score[i] = pos[i].copy(), score
            if score < gbest_score:             # update global best
                gbest, gbest_score = pos[i].copy(), score
    if (it + 1) % 10 == 0:
        print(f"Iteration {it + 1:3d}: best SSE = {gbest_score:.2f}")

labels = np.argmin(np.linalg.norm(X[:, None, :] - gbest[None, :, :], axis=2), axis=1)
print("\nFinal centroids:\n", np.round(gbest, 2))
print(f"Final SSE: {gbest_score:.2f}")

from sklearn.cluster import KMeans
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
print(f"K-Means SSE (for comparison): {km.inertia_:.2f}")

plt.scatter(X[:, 0], X[:, 1], c=labels, s=15)
plt.scatter(gbest[:, 0], gbest[:, 1], c='red', marker='X', s=200)
plt.title('PSO Clustering'); plt.show()
