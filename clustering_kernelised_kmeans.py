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
n = 100
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
#  exp{−γ * (xn − xm).T @ (xn − xm)}
def gaussian_kernel(gamma=1):
    def kernel(X_n, X_m):
        diff = X_n[:, None, :] - X_m
        sq_diff = (diff**2).sum(axis=-1)
        return np.exp(-gamma * sq_diff).squeeze()

    return kernel


# xn.T @ xm
def linear_kernel(gamma=None):
    def kernel(X_n, X_m):
        dot_products = X_n[:, None, :] @ X_m.T
        return dot_products.squeeze()

    return kernel


# (1 + xn.T @ xm)**γ
def polinomial_kernel(gamma=1):
    def kernel(X_n, X_m):
        dot_products = X_n[:, None, :] @ X_m.T
        polynomials = (1 + dot_products)**gamma
        return polynomials.squeeze()

    return kernel


# %%
def create_assignment_matrix(K, assignments):
    N = len(assignments)
    assignment_matrix = np.zeros((N, K, 1))
    assignment_matrix[range(N), assignments] = 1
    return assignment_matrix


def kmeans_kernelised_distance(
        K,
        X_n,
        num_iter=10,
        kernel_function=linear_kernel(),
):
    assignments = np.random.randint(0, K, size=len(X_n))
    assignment_matrix = create_assignment_matrix(K, assignments).squeeze()
    mu_k = X_n[np.random.randint(0, len(X_n), size=K)]
    counter = Counter({k: 0 for k in range(K)})
    counter.update(Counter(np.sort(assignments)))
    losses = np.zeros(num_iter)
    X_K_distances = np.zeros((len(X_n), K))
    for i in range(num_iter):

        cnts = np.array(list(counter.values()))
        #  K(x_n; x_n)
        k_X_X = np.diag(kernel_function(X_n, X_n))[:, None].squeeze()

        for k in range(K):
            idx_of_members = assignments == k
            if not any(idx_of_members):
                continue
            N_k = np.sum(idx_of_members)
            z_m = idx_of_members[:, None]
            z_r = idx_of_members[:, None]
            k_X_mu = (2 / N_k) * (z_m.T *
                                  kernel_function(X_n, X_n)).sum(axis=1)
            k_X_mu_mu = (1 / (N_k**2)) * np.sum(
                (z_m @ z_r.T) * kernel_function(X_n, X_n))
            k_dist_X2X = k_X_X - k_X_mu + k_X_mu_mu
            X_K_distances[:, k] = k_dist_X2X
            mu_k[k] = X_n[idx_of_members].mean(axis=0)
            loss_k = ((X[idx_of_members][:, None, :] - mu_k[k][None, :])**2)
            loss_k = loss_k.sum(axis=-1)
            loss_k = np.sqrt(loss_k)
            loss_k = loss_k.sum()
            losses[i] += loss_k
        assignments = X_K_distances.argmin(axis=-1)

    return mu_k, assignments, losses


centroids_mah, assigments_mah, losses_mah = kmeans_kernelised_distance(
    5, X, kernel_function=gaussian_kernel())

plt.plot(losses_mah[::2])
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
plot_clustering(X, assigments_mah, ax, centroids_mah)
for mean, cov in zip(X_means, X_covs):
    plot_contour_2d(mean, cov, ax)

# %%

centroids_euc, assigments_euc, losses_euc = kmeans_kernelised_distance(
    4, X, kernel_function=polinomial_kernel())

plt.plot(losses_euc[::2])
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
plot_clustering(X, assigments_euc, ax, centroids_euc)
for mean, cov in zip(X_means, X_covs):
    plot_contour_2d(mean, cov, ax)
# %%

centroids_euc, assigments_euc, losses_euc = kmeans_kernelised_distance(
    4, X, kernel_function=gaussian_kernel())

plt.plot(losses_euc[::2])
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
plot_clustering(X, assigments_euc, ax, centroids_euc)
for mean, cov in zip(X_means, X_covs):
    plot_contour_2d(mean, cov, ax)
# %%
num_rows = 5
fig, axes = plt.subplots(num_rows, 3, sharey=True, sharex=True, figsize=(15, num_rows*5))
faxes = axes.flatten()
K = 4
for rnum, ax_row in enumerate(axes):
    ax = ax_row[0]
    centroids, assigments, losses = kmeans_kernelised_distance(
        K, X, kernel_function=linear_kernel())
    plot_clustering(X, assigments, ax)
    for mean, cov in zip(X_means, X_covs):
        plot_contour_2d(mean, cov, ax)
    ax.legend()
    ax.set_title(f"Linear Kernel ")
    ax = ax_row[1]
    centroids, assigments, losses = kmeans_kernelised_distance(
        K, X, kernel_function=polinomial_kernel(rnum))
    plot_clustering(X, assigments, ax)
    for mean, cov in zip(X_means, X_covs):
        plot_contour_2d(mean, cov, ax)
    ax.legend()
    ax.set_title(f"{rnum}-order Polynomial Kernel")
    ax = ax_row[2]
    centroids, assigments, losses = kmeans_kernelised_distance(
        K, X, kernel_function=gaussian_kernel(1/(num_rows-rnum)))
    plot_clustering(X, assigments, ax)
    for mean, cov in zip(X_means, X_covs):
        plot_contour_2d(mean, cov, ax)
    ax.legend()
    ax.set_title(f"Gauss Kernel with γ={1}/{num_rows-rnum}")

fig.tight_layout()
plt.show()
# %%
num_rows = 5
fig, axes = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(15, 15))
faxes = axes.flatten()
K = 4

for rnum in range(5):
    ax = axes[0]
    centroids, assigments, losses = kmeans_kernelised_distance(
        K, X, kernel_function=linear_kernel())
    ax.plot(losses, label="Linear Kernel")
    ax.legend()    
    ax.set_title(f"Linear Kernel")
    ax = axes[1]
    centroids, assigments, losses = kmeans_kernelised_distance(
        K, X, kernel_function=polinomial_kernel(rnum))
    ax.plot(losses, label=f"{rnum}-order Polynomial Kernel")
    ax.legend()    
    ax.set_title(f"Polynomial Kernel")
    ax = axes[2]
    centroids, assigments, losses = kmeans_kernelised_distance(
        K, X, kernel_function=gaussian_kernel(1/(num_rows-rnum)))
    ax.plot(losses, label=f"Gauss Kernel with γ={1}/{num_rows-rnum}")
    ax.legend()    
    ax.set_title(f"Gauss Kernel")
    

fig.tight_layout()
plt.show()