# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
import scipy
import random as r
from data import observed_data, true_function
from helper import add_bias_vector, create_polinomial_bases

# %%
n = 100
val_n = 10
data = np.vstack(np.array(observed_data(n, 2, 3)).T)
val_data = np.vstack(np.array(observed_data(val_n)).T)
display(data.max(axis=0))
display(data.min(axis=0))
display(data.mean(axis=0))

# %%
# %%
X, y = data[:, :-1], data[:, -1]
train_X = add_bias_vector(create_polinomial_bases(X))
train_X.shape
# %%


def compute_weights(X, y, lmb=0):
    w = np.linalg.inv(X.T @ X + lmb * np.eye(X.shape[1])) @ X.T @ y
    return w


w_pred = compute_weights(train_X, y).reshape(-1, 1)
w_pred
# %%

grid_X = np.repeat([np.linspace(-5, 5, 100)], 2, axis=0).T
plot_X = add_bias_vector(create_polinomial_bases(grid_X))

xs = grid_X[:, 0]
ys = grid_X[:, 1]
zs = plot_X @ w_pred
zs.shape

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(xs, ys, zs.squeeze(), c='red')
ax.scatter(data[:, 0], data[:, 1], data[:, 2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
# %%

# %%
