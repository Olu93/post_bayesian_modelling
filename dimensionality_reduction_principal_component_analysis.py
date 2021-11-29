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
from data import observed_data_classification, observed_data_classification_two_features
from helper import add_bias_vector, compute_metrics, create_polinomial_bases, predict, sigmoid
from viz import plot_clustering, plot_contour_2d, plot_countours_and_points, plot_train_val_curve, plot_w_samples
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics
from sklearn import model_selection
from tqdm.notebook import tqdm

# %%
data = datasets.load_wine(return_X_y=True, as_frame=True)
data_X, data_y = data[0], data[1]
data_X
# %%
normed_X = (data_X - data_X.mean()) / data_X.std()
normed_X


# %%
def principal_component_analysis(X, dim=3):
    N = len(X)
    C = (X.values.T @ X.values) / N
    eigvals, eigvecs = np.linalg.eig(C)
    n_highest_eigvecs = eigvecs[:, :dim]
    projection = X @ n_highest_eigvecs
    return projection


fig = plt.figure(figsize=(15, 7))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122, projection="3d")
projected_data = principal_component_analysis(normed_X, dim=3)
projected_data = projected_data.join(data_y)

for cls in data_y.unique():
    members = projected_data.iloc[:, 3] == cls
    c_projected_data = projected_data.loc[members]
    ax1.scatter(
        x=c_projected_data.iloc[:, 0],
        y=c_projected_data.iloc[:, 1],
        # c=c_projected_data.iloc[:, 3],
        label=f"Class {cls}",
    )
    ax1.legend()
    ax2.scatter(
        xs=c_projected_data.iloc[:, 0],
        ys=c_projected_data.iloc[:, 1],
        zs=c_projected_data.iloc[:, 2],
        # c=c_projected_data.iloc[:, 3],
        label=f"Class {cls}",
    )
    ax2.legend()
    ax2.view_init(30, 45)
plt.show()
Axes3D