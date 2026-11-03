# Assignment 1: Fuzzy set operations, Cartesian product, Max-Min composition
import numpy as np

# Two fuzzy sets on the same universe X = {x1, x2, x3}
A = np.array([0.2, 0.7, 1.0])
B = np.array([0.5, 0.4, 0.9])
print("A =", A)
print("B =", B)

# ---------- Fuzzy set operations ----------
print("\nUnion (max)          :", np.maximum(A, B))
print("Intersection (min)   :", np.minimum(A, B))
print("Complement of A (1-A):", np.round(1 - A, 2))
print("Complement of B (1-B):", np.round(1 - B, 2))
print("Difference A-B       :", np.round(np.minimum(A, 1 - B), 2))
print("Algebraic sum        :", np.round(A + B - A * B, 2))
print("Algebraic product    :", np.round(A * B, 2))
print("Bounded sum          :", np.round(np.minimum(1, A + B), 2))
print("Bounded difference   :", np.round(np.maximum(0, A - B), 2))

# De Morgan's law check: (A U B)' = A' ∩ B'
lhs = 1 - np.maximum(A, B)
rhs = np.minimum(1 - A, 1 - B)
print("De Morgan's law holds:", np.allclose(lhs, rhs))

# ---------- Fuzzy relation by Cartesian product ----------
# R(x, y) = min(A(x), B(y))
P = np.array([0.3, 0.8])          # fuzzy set on X = {x1, x2}
Q = np.array([0.5, 0.9, 0.2])     # fuzzy set on Y = {y1, y2, y3}
R = np.minimum.outer(P, Q)
print("\nRelation R = P x Q:\n", R)

# Second relation S on Y x Z
S = np.array([[0.6, 0.4],
              [0.7, 0.9],
              [0.3, 1.0]])
print("\nRelation S:\n", S)

# ---------- Max-Min composition ----------
# T(x, z) = max over y of min(R(x, y), S(y, z))
T = np.zeros((R.shape[0], S.shape[1]))
for i in range(R.shape[0]):
    for j in range(S.shape[1]):
        T[i, j] = np.max(np.minimum(R[i, :], S[:, j]))
print("\nMax-Min composition T = R o S:\n", T)
