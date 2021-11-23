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
def gaussian_kernel(X_n, X_m):
    pass

def linear_kernel(X_n, X_m):
    dot_products = X_n[:, None, :] @ X_m.T
    return dot_products.squeeze()    
    
def polinomial_kernel(X_n, X_m):
    pass

# %%
def kmeans_kernelised_distance(K, X, num_iter=10, kernel_function = linear_kernel):
    assignments = np.random.randint(0, K, size=len(X))
    mu_k = X[np.random.randint(0, len(X), size=K)]
    counter = Counter({k:0 for k in range(K)})
    counter.update(Counter(np.sort(assignments)))
    losses = np.zeros(num_iter)
    for i in range(num_iter):

        cnts = np.array(list(counter.values()))[None, :]
        #  K(x_n; x_n)
        k_X_X = np.diag(kernel_function(X, X))[:, None]
        # z_m_k * K(x_n; x_m)
        k_X_mu = kernel_function(X, mu_k)
        normed_k_X_mu = 2 * (k_X_mu/cnts)

        # k_mu_mu = kernel_function(mu_k, mu_k)
        normed_k_mu_mu = np.diag(1 * (k_mu_mu/(cnts**2)))[None, :]

        
        dist_X2X_per_class = k_X_X - normed_k_X_mu + normed_k_mu_mu
        assignments = dist_X2X_per_class.argmin(axis=-1)
        for k in range(K):
            idx_of_members = assignments == k
            if not any(idx_of_members):
                continue
            mu_k[k] = X[idx_of_members].mean(axis=0)
            loss_k = ((X[idx_of_members][:, None, :] - mu_k[k][None, :])**2)
            loss_k = loss_k.sum(axis=-1)
            loss_k = np.sqrt(loss_k)
            loss_k = loss_k.sum()
            losses[i] += loss_k

    return mu_k, assignments, losses


centroids_mah, assigments_mah, losses_mah = kmeans_kernelised_distance(4, X)

plt.plot(losses_mah[::2])
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
plot_clustering(X, assigments_mah, ax, centroids_mah)
for mean, cov in zip(X_means, X_covs):
    plot_contour_2d(mean, cov, ax)
# %%
def kmeans_eucledian_distance(K, X, num_iter=10):
    assignments = np.random.randint(0, K, size=len(X))
    mu_k = X[np.random.randint(0, len(X), size=K)]
    losses = np.zeros(num_iter)
    for i in range(num_iter):
        X_differences = X[:, None, :] - mu_k
        X_sq_differences = X_differences**2
        X_sum_sq_differences = X_sq_differences.sum(axis=-1)
        X_root_sum_sq_differences = np.sqrt(X_sum_sq_differences)
        assignments = X_root_sum_sq_differences.argmin(axis=-1)
        for k in range(K):
            idx_of_members = assignments == k
            if not any(idx_of_members):
                continue
            mu_k[k] = X[idx_of_members].mean(axis=0)
            loss_k = ((X[idx_of_members][:, None, :] - mu_k[k][None, :])**2)
            loss_k = loss_k.sum(axis=-1)
            loss_k = np.sqrt(loss_k)
            loss_k = loss_k.sum()
            losses[i] += loss_k

    return mu_k, assignments, losses


centroids_euc, assigments_euc, losses_euc = kmeans_eucledian_distance(4, X)

plt.plot(losses_euc[::2])
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
plot_clustering(X, assigments_euc, ax, centroids_euc)
for mean, cov in zip(X_means, X_covs):
    plot_contour_2d(mean, cov, ax)
# %%

fig, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(10, 10))
ax.plot(losses_euc[::2], label="Eucledian Distance")
ax.plot(losses_mah[::2], label="Mahalanobis Distance")
ax.legend()
fig.tight_layout()
plt.show()