# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import random as r
# %%
np.random.seed(42)


def true_function(x, y, p1, p2):
    return (3 * (x**p1)) - ((1 / 2) * y**p2) - 1


def true_function_linear(x, p1=1):
    return (3 * (x**p1)) - 1


def observed_data(d: int = 10, p1=2, p2=3):
    data = np.random.randn(d, 2) * 3
    x, y = data[:, 0], data[:, 1]
    variance = 2.5
    return x, y, true_function(x, y, p1, p2) + np.random.randn(d) * variance


def observed_data_linear(d: int = 10, p1=1):
    data = np.random.randn(d, 2) * 3
    x, y = data[:, 0], data[:, 1]
    return x, true_function_linear(x, p1) + np.random.randn(d)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 50

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs, ys, zs = observed_data(n)
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
# %%
