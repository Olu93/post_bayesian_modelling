import numpy as np
from scipy import special

STABILITY_CONSTANT = 0.00000000001


def log_stable(x):
    return np.log(x + STABILITY_CONSTANT)


def sigmoid(x):
    return special.expit(x)


def add_bias_vector(X):
    return np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])


def create_polinomial_bases(X, order=3):
    return np.hstack([X**i for i in range(1, order + 1)])


def predict(ws, X):
    logits = X @ ws.T
    probabilities = sigmoid(logits).mean(axis=1)
    return probabilities[:, None]


def compute_metrics(w, X, y):
    y_hat = sigmoid(X @ w)
    losses = y - y_hat
    m_loss = np.mean(np.abs(losses))
    m_acc = np.mean(y == ((y_hat >= 0.5) * 1.0))
    return m_loss, m_acc
