import numpy as np
from scipy import special


def sigmoid(x):
    return special.expit(x)

def add_bias_vector(X):
    return np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])


def create_polinomial_bases(X, order=3):
    return np.hstack([X**i for i in range(1, order + 1)])