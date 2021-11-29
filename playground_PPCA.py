# %%
from sklearn import datasets
import abc
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
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.5g" % x))

# %%
prior_means = np.array([1, 4, 10, 1, 100]) 
D = len(prior_means)
M = 100
N = 42
X = np.random.normal(size=(N, D))
exp_W = np.random.multivariate_normal(prior_means,
                                      np.eye(D),
                                      size=M)
# %%
# DxN @ NxD
projected_X = X.T@X
# DxD
projected_X/N

# %%
mean_W = exp_W.mean(0)
# Dx1 @ 1xD 
cov_W = (mean_W[:, None]@mean_W[:, None].T)
display(cov_W)
# %%

# %%
