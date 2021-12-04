# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.lib.function_base import meshgrid
from numpy.random import multivariate_normal
import pandas as pd
from IPython.display import display
from scipy import stats
from matplotlib import cm
import random as r
from data import observed_data, observed_data_linear, observed_data_wobbly, true_function_polynomial
from helper import add_bias_vector, create_polinomial_bases
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import Delaunay
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
# %%
n = 500
exp_1, exp_2 = 2, 3
val_n = 50
p_ord = 2
data = np.vstack(np.array(observed_data_wobbly(n, exp_1, exp_2, 0)).T)
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(data[:,0],data[:,1], data[:, 2])

# %%