import numpy as np

class MatrixCalculator:
    @staticmethod
    def add(A, B):
        return A + B

    @staticmethod
    def subtract(A, B):
        return A - B

    @staticmethod
    def multiply(A, B):
        return A @ B

    @staticmethod
    def transpose(A):
        return A.T

    @staticmethod
    def determinant(A):
        return np.linalg.det(A)

    @staticmethod
    def inverse(A):
        return np.linalg.inv(A)

    @staticmethod
    def rank(A):
        return np.linalg.matrix_rank(A)

    @staticmethod
    def eigen(A):
        vals, vecs = np.linalg.eig(A)
        return vals, vecs

    @staticmethod
    def solve(A, b):
        return np.linalg.solve(A, b)

# Demo
A = np.array([[2, 1], [5, 3]], dtype=float)
B = np.array([[1, 4], [2, 0]], dtype=float)
b = np.array([7, 19], dtype=float)

print("A + B:\n", MatrixCalculator.add(A, B))
print("A @ B:\n", MatrixCalculator.multiply(A, B))
print("det(A):", MatrixCalculator.determinant(A))
print("inv(A):\n", MatrixCalculator.inverse(A))
print("rank(A):", MatrixCalculator.rank(A))
eigvals, eigvecs = MatrixCalculator.eigen(A)
print("eigenvalues:", eigvals)
print("solve Ax=b, x:", MatrixCalculator.solve(A, b))
