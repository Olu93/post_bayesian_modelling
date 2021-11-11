# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import random as r
# %%
np.random.seed(42)


def true_function_polynomial(x, y, p1, p2):
    return (3 * (x**p1)) - ((1 / 2) * y**p2) - 1


def d1_function_polynomial(x, y, p1, p2):
    dx = (p1) * 3 * x**(p1 - 1)
    dy = (p2) * (1 / 2) * y**(p2 - 1)

    return dx, dy

def opt_d1_function_polynomial(x, y, p1, p2):
    dx_opt = np.power(x, 1/(p1 - 1))-x
    dy_opt = np.power(y, 1/(p2 - 1))-y

    return dx_opt, dy_opt


def d2_function_polynomial(x, y, p1, p2):
    dxx = (p1) * (p1 - 1) * 3 * x**(p1 - 2)
    dxy = np.zeros_like(dxx)
    dyy = (p2) * (p2 - 1) * (1 / 2) * y**(p2 - 2)
    dyx = np.zeros_like(dyy)

    return np.array([[dxx, dxy], [dyx,dyy]])


def true_function_linear(x, p1=1):
    return (3 * (x**p1)) - 1


def true_function_sigmoid(x, y, w1, w2):
    return 1 / (1 + np.exp(-(w1 * x + w2 * y)))


def observed_data(d: int = 10, p1=2, p2=3):
    data = np.random.randn(d, 2) * 3
    x, y = data[:, 0], data[:, 1]
    variance = 2.5
    return x, y, true_function_polynomial(x, y, p1, p2) + np.random.randn(d) * variance


def observed_data_wobbly(d: int = 10):
    data = np.random.randn(d, 2) * 1
    x, y = data[:, 0], data[:, 1]
    variance = 2.5
    return x, y, ((np.sin(x) * 5) / np.exp(y)) + np.random.randn(d) * variance


def observed_data_linear(d: int = 10, p1=1):
    data = np.random.randn(d, 2) * 3
    x, y = data[:, 0], data[:, 1]
    return x, true_function_linear(x, p1) + np.random.randn(d)


def observed_data_binary(d: int = 10, w1=2, w2=2, std=3):
    data = np.random.randn(d, 2) * std
    x, y = data[:, 0], data[:, 1]
    return x, y, true_function_sigmoid(x, y, w1, w2) > 0.5


# %%
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
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 50

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs, ys, zs = observed_data_wobbly(n)
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 50

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
xs, ys, zs = observed_data_binary(n)
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()