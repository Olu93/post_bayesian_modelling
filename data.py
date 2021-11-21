# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import random as r

from scipy import special

from helper import sigmoid
from viz import plot_contour_2d
# https://github.com/jkclem/MCMC-Log-Reg/blob/master/MCMC%20Logistic%20Regression.ipynb
# %%
np.random.seed(42)


def true_function_polynomial(x, y, p1, p2):
    return (3 * (x**p1)) - ((1 / 2) * y**p2) - 1


def d1_function_polynomial(x, y, p1, p2):
    dx = (p1) * 3 * x**(p1 - 1)
    dy = (p2) * (1 / 2) * y**(p2 - 1)

    return dx, dy


def opt_d1_function_polynomial(x, y, p1, p2):
    dx_opt = np.power(x, 1 / (p1 - 1)) - x
    dy_opt = np.power(y, 1 / (p2 - 1)) - y

    return dx_opt, dy_opt


def d2_function_polynomial(x, y, p1, p2):
    dxx = (p1) * (p1 - 1) * 3 * x**(p1 - 2)
    dxy = np.zeros_like(dxx)
    dyy = (p2) * (p2 - 1) * (1 / 2) * y**(p2 - 2)
    dyx = np.zeros_like(dyy)

    return np.array([[dxx, dxy], [dyx, dyy]])


def true_function_linear(x, p1=1):
    return (3 * (x**p1)) - 1


def true_function_sigmoid(x, y, w1, w2):
    x = (w1 * x + w2 * y)
    return sigmoid(x)


def true_function_softmax(X, W):
    # Compute all single logits per data point and class
    logits = X[None, :, :] @ W.T
    # Compute singular probits
    probits = sigmoid(logits)
    # Compute sum of class distributions per data point
    sum_of_probits = probits.sum(axis=-1)
    # Compute softmax by dividing row wise
    probabilities_per_class = probits / sum_of_probits.T
    return probabilities_per_class.squeeze()


def observed_data(d: int = 10, p1=2, p2=3):
    data = np.random.randn(d, 2) * 3
    x, y = data[:, 0], data[:, 1]
    variance = 2.5
    return x, y, true_function_polynomial(x, y, p1,
                                          p2) + np.random.randn(d) * variance


def observed_data_wobbly(d: int = 10):
    data = np.random.randn(d, 2) * 1
    x, y = data[:, 0], data[:, 1]
    variance = 2.5
    return x, y, ((np.sin(x) * 5) / np.exp(y)) + np.random.randn(d) * variance


def observed_data_linear(d: int = 10, p1=1):
    data = np.random.randn(d, 2) * 3
    x, y = data[:, 0], data[:, 1]
    return x, true_function_linear(x, p1) + np.random.randn(d)


def observed_data_binary(d: int = 10, w1=2, w2=2, std=3, noise=0):
    data = np.random.normal(0, std, size=(d, 2))
    # data = np.random.uniform(-std, std, size=(d, 2))
    # print(data)
    x, y = data[:, 0], data[:, 1]
    probability = true_function_sigmoid(x, y, w1, w2)
    err = np.random.randn(d) * noise
    probability = probability + err
    z = (probability >= 0.5) * 1
    return x, y, z


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
ax = fig.add_subplot(1, 2, 1, projection='3d')

n = 50
w1, w2 = 0.1, -0.7
xs, ys, zs = observed_data_binary(n, w1, w2, std=50)
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(30, 15)

ax = fig.add_subplot(1, 2, 2, projection='3d')

zs = true_function_sigmoid(xs, ys, w1, w2)
ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(30, 15)
plt.show()

# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.5g" % x))


def observed_data_classification(num_data,
                                 spread_data,
                                 num_weights=5,
                                 num_classes=3,
                                 noisy=False,
                                 seed=42):
    np.random.seed(seed)
    class_weights = np.random.normal(size=(num_classes, num_weights))
    # X = np.random.normal(0, spread_data, size=(num_data, num_weights))
    X = np.random.uniform(-spread_data,
                          spread_data,
                          size=(num_data, num_weights))
    P_c_X = true_function_softmax(X, class_weights)
    y = np.argmax(P_c_X, axis=-1)
    if noisy:
        c = P_c_X.cumsum(axis=-1)
        u = np.random.rand(len(c), 1)
        y = (u <= c).argmax(axis=-1)

    return X, y[:, None], class_weights, P_c_X


X, y, W_c, P_c = observed_data_classification(1000, 100, 5, 10, 1, 42)
plt.hist(y, bins=len(np.unique(y)))
plt.show()
plt.hist(P_c[:, 3])
plt.show()


# %%
def observed_data_classification_two_features(num_data,
                                              mean_data,
                                              cov_data,
                                              num_classes=3,
                                              noisy=False,
                                              seed=42):
    np.random.seed(seed)
    num_weights = len(mean_data)
    class_weights = np.random.normal(size=(num_classes, num_weights))
    X = np.random.multivariate_normal(mean_data, cov_data, size=num_data)
    P_c_X = true_function_softmax(X, class_weights)
    y = np.argmax(P_c_X, axis=-1)
    if noisy:
        c = P_c_X.cumsum(axis=-1)
        u = np.random.rand(len(c), 1)
        y = (u <= c).argmax(axis=-1)

    return X, y[:, None], class_weights, P_c_X


cov_data = np.array([[10, 8], [8, 10]])
mean_data = np.array([5, 5])
X, y, W_c, P_c = observed_data_classification_two_features(
    1000, mean_data, cov_data, 10, 1, 42)
plt.hist(y, bins=len(np.unique(y)))
plt.show()
plt.hist(P_c[:, 3])
plt.show()
ax = plt.gca()
plot_contour_2d(mean_data, cov_data, ax)
# %%

# %%
