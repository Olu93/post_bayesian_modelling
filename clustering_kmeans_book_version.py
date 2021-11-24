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
from data import observed_data_classification, observed_data_classification_two_features
from helper import add_bias_vector, compute_metrics, create_polinomial_bases, predict, sigmoid
from viz import plot_clustering, plot_contour_2d, plot_countours_and_points, plot_train_val_curve, plot_w_samples
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics
from sklearn import model_selection
from tqdm.notebook import tqdm
# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True

# %%
n = 200
n_c = 3
x1mu, x2mu = 1, 1
x1std, x2std = 3, 3
xystd = 1
val_n = 100
noise = True
seed = 42
cov_data = np.array([[x1std, xystd], [xystd, x2std]])
mean_data = np.array([x1mu, x2mu])

X, true_y, X_means, X_covs = observed_data_classification_two_features(
    n, mean_data, cov_data)

# %%

ax = plt.gca()
plot_countours_and_points(ax, X, true_y, X_means, X_covs)
plt.show()


# %%
# Follows: Rogers, Simon, and Mark Girolami. A First Course in Machine Learning, Second Edition. 2nd ed. Chapman & Hall/CRC, 2016. https://nl1lib.org/book/11571143/9a0735.
def create_assignment_matrix(K, N, assignments):
    assignment_matrix = np.zeros((N, K, 1))  
    assignment_matrix[range(N), assignments] = 1
    return assignment_matrix

def compute_mus(K, X, assignment_matrix):
    # assignment_matrix: N x K x 1
    # N x K x X_ft
    X_K = np.repeat(X[:, None, :], K, axis=1)  
    new_mu = (assignment_matrix * X_K).sum(axis=0) / assignment_matrix.sum(
        axis=0)
    return new_mu

def compute_euc_distance(X, mu_k):
    X_differences = X[:, None, :] - mu_k
    X_sq_differences = X_differences**2
    X_sum_sq_differences = X_sq_differences.sum(axis=-1)
    X_root_sum_sq_differences = np.sqrt(X_sum_sq_differences)
    return X_root_sum_sq_differences


def kmeans_eucledian_distance(K, X, num_iter=10):
    N = len(X)
    assignments = np.random.randint(0, K, size=len(X))
    assignment_matrix = create_assignment_matrix(K, N, assignments)
    mu_k = compute_mus(K, X, assignment_matrix)
    losses = np.zeros(num_iter)
    for i in range(num_iter):
        # N x K
        least_squares = compute_euc_distance(X, mu_k)
        assignments = least_squares.argmin(axis=-1)
        # N x K x 1
        assignment_matrix = create_assignment_matrix(K, N, assignments)
        new_mus = compute_mus(K, X, assignment_matrix)
        mu_k[~np.isnan(new_mus)] = new_mus[~np.isnan(new_mus)]
        losses[i] = (least_squares * assignment_matrix.squeeze()).sum()

    return mu_k, assignments, losses


centroids_euc, assigments_euc, losses_euc = kmeans_eucledian_distance(5, X)

# centroids_euc
plt.plot(losses_euc[::2])

# %%

fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
plot_clustering(X, assigments_euc, ax, centroids_euc)
for mean, cov in zip(X_means, X_covs):
    plot_contour_2d(mean, cov, ax)
