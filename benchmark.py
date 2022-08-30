import numpy as np


"""Separable functions:"""

"""[-100, 100]"""
def Sphere(X):
    result = 0
    for x in X:
        result += x ** 2
    return result * (1 + np.random.normal(loc=0, scale=0.01, size=None))


"""[-5.12, 5.12]"""
def Rastrigin(X):
    result = 10 * len(X)
    for x in X:
        result += x ** 2 - 10 * np.cos(2 * np.pi * x)
    return result * (1 + np.random.normal(loc=0, scale=0.01, size=None))


"""Non-separable functions:"""
"""[-30, 30]"""
def Rosenbrock(X):
    result = 0
    Dim = len(X)
    for i in range(Dim-1):
        result += 100 * (X[i+1] - X[i] ** 2) ** 2 + (X[i] - 1) ** 2
    return result * (1 + np.random.normal(loc=0, scale=0.01, size=None))


"""[-32.768, 32.768]"""
def Ackley(X):
    p1 = 0
    p2 = 0
    for x in X:
        p1 += x ** 2
        p2 += np.cos(2 * np.pi * x)
    p1 = -0.2 * np.sqrt(1/len(X) * p1)
    p2 = p2 / len(X)
    return (-20 * np.exp(p1) - np.exp(p2) + 20 + np.e) * (1 + np.random.normal(loc=0, scale=0.01, size=None))


"""[-10, 10]"""
def Dixon_Price(X):
    result = (X[0] - 1) ** 2
    Dim = len(X)
    for i in range(1, Dim):
        result += i * (2 * X[i] ** 2 - X[i-1]) ** 2
    return result * (1 + np.random.normal(loc=0, scale=0.01, size=None))