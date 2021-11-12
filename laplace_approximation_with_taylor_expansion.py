# %%
import abc
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import meshgrid
from numpy.random import multivariate_normal
import pandas as pd
from IPython.display import display
from scipy import stats
from matplotlib import cm
import random as r
from data import d1_function_polynomial, d2_function_polynomial, observed_data, observed_data_binary, observed_data_linear, opt_d1_function_polynomial, true_function_polynomial
from helper import add_bias_vector, create_polinomial_bases
from tqdm import tqdm

# %%
# True Function Data
exp1, exp2 = 3, 4
width = 1
x = np.linspace(-width, width, 100)
y = np.linspace(-width, width, 100)

X, Y = np.meshgrid(x, y)
Z = true_function_polynomial(X, Y, exp1, exp2)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
# %%
# First Derivatives
X, Y = np.meshgrid(x, y)
DX, DY = d1_function_polynomial(X, Y, exp1, exp2)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, DX, alpha=0.75)
ax.plot_surface(X, Y, DY, alpha=0.75)
# %%
# Second Derivatives
X, Y = np.meshgrid(x, y)
DXY_COV = d2_function_polynomial(X, Y, exp1, exp2)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, DXY_COV[0, 0], alpha=0.75)
ax.plot_surface(X, Y, DXY_COV[0, 1], alpha=0.75)
ax.plot_surface(X, Y, DXY_COV[1, 0], alpha=0.75)
ax.plot_surface(X, Y, DXY_COV[1, 1], alpha=0.75)

# %%

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(x, y)
Z = true_function_polynomial(X, Y, exp1, exp2)
D0x, D0y = opt_d1_function_polynomial(X, Y, exp1, exp2)
point1, point2 = D0x, D0y
DXY_COV = d2_function_polynomial(D0x, D0y, exp1, exp2)
ax.plot_surface(X, Y, Z, alpha=0.25)
ax.set_xlabel = ("X")
ax.set_ylabel = ("Y")
D0X, D0Y = np.meshgrid(x, y)
ax.scatter(point1, D0Y, true_function_polynomial(point1, D0Y, exp1, exp2), s=10, c="red", alpha=0.1)
ax.scatter(D0X, point2, true_function_polynomial(D0X, point2, exp1, exp2), s=10, c="blue", alpha=0.1)
ax.scatter(D0x, D0y, true_function_polynomial(D0x, D0y, exp1, exp2), s=100, marker="o", c="green")
ax.view_init(30, 70)
fig.tight_layout()
plt.show()


# %%
def taylor_approximation_2nd_order(x, y, exp1, exp2):
    X, Y = np.meshgrid(x, y)
    f_XY_both = true_function_polynomial(X, Y, exp1, exp2)
    d2f_XY = d2_function_polynomial(X, Y, exp1, exp2)
    f_XX = d2f_XY[0, 0]
    f_XY = d2f_XY[0, 1]
    f_YX = d2f_XY[1, 0]
    f_YY = d2f_XY[1, 1]
    new_f_XX = f_XY_both / 1 + ((f_XX / 2) * (X * X))
    new_f_XY = f_XY_both / 1 + ((f_XY / 2) * (X * Y))
    new_f_YX = f_XY_both / 1 + ((f_YX / 2) * (Y * X))
    new_f_YY = f_XY_both / 1 + ((f_YY / 2) * (Y * Y))
    return np.array([[new_f_XX, new_f_XY], [new_f_YX, new_f_YY]])


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
vals = taylor_approximation_2nd_order(x, y, exp1, exp2)
ax1.plot_surface(X, X, vals[0, 0], alpha=0.5, label="XX")
ax1.plot_surface(X, Y, vals[0, 1], alpha=0.5, label="XY")
ax1.plot_surface(Y, X, vals[1, 0], alpha=0.5, label="YX")
ax1.plot_surface(Y, Y, vals[1, 1], alpha=0.5, label="YY")
mid = len(X) // 2
zeros = np.zeros(len(X))
# ax1.plot(X[mid,:], Y[:, mid], vals[0, 1][mid,:], alpha=0.5, label="TrueXY", c='black')
# ax1.scatter(X.flatten(), 0, vals[0, 1], alpha=0.5, label="TrueXY", c='black')
# ax1.scatter(0, Y.flatten(), vals[1, 0], alpha=0.5, label="TrueXY", c="black")
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(X, Y, true_function_polynomial(X, Y, exp1, exp2), alpha=0.5, label="TrueXY")
ax2.scatter(X.flatten(), 0, true_function_polynomial(X.flatten(), 0, exp1, exp2), alpha=0.5, label="TrueXY")
ax2.scatter(0, Y.flatten(), true_function_polynomial(0, Y.flatten(), exp1, exp2), alpha=0.5, label="TrueXY")
ax1.set_zlim3d(ax2.get_zlim3d())
# ax.legend()
fig.tight_layout()
plt.show()
# ax.plot(X,Y,vals[1])