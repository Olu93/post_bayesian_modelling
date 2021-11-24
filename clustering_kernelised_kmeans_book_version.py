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
# Follows: Rogers, Simon, and Mark Girolami. A First Course in Machine Learning, Second Edition. 2nd ed. Chapman & Hall/CRC, 2016. https://nl1lib.org/book/11571143/9a0735.
def compute_euc_distance(X, mu_k):
    X_differences = X[:, None, :] - mu_k
    X_sq_differences = X_differences**2
    X_sum_sq_differences = X_sq_differences.sum(axis=-1)
    X_root_sum_sq_differences = np.sqrt(X_sum_sq_differences)
    return X_root_sum_sq_differences


def create_assignment_matrix(K, assignments):
    N = len(assignments)
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

def kmeans_kernelised_distance(
        K,
        X_n,
        num_iter=10,
        kernel_function=linear_kernel(),
):
    assignments = np.random.randint(0, K, size=len(X_n))
    # N x K x 1
    assignment_matrix = create_assignment_matrix(K, assignments)
    mu_k = X_n[np.random.randint(0, len(X_n), size=K)]
    counter = Counter({k: 0 for k in range(K)})
    counter.update(Counter(np.sort(assignments)))
    losses = np.zeros(num_iter)
    for i in range(num_iter):

        cnts = np.array(list(counter.values()))[:, None]
        
        # K x N --> Real: K x 1 x N
        z_cm = assignment_matrix.squeeze().T[:, None, :]
        # N x N --> Real: N x N
        c_X_X = kernel_function(X_n, X_n)
        # N --> Real: N x 1
        knn = np.diag(c_X_X)[None, :]
        
        # K x N x N 
        z_cmK = z_cm * c_X_X
        # K x N
        sum_z_cmK = z_cmK.sum(axis=-1)
        # K x N
        normed_sum_z_cK = (2/cnts) * sum_z_cmK 

        # K x N --> Real: K x N x 1
        z_kr = np.transpose(z_cm, axes=(0,2,1))
        # K x N x N 
        z_cmcrK = z_cm * z_kr * c_X_X
        # K --> Real: K x 1
        sum_z_cmcrK = z_cmcrK.sum(axis=(-1,-2))[:, None]
        # K x 1
        normed_sum_z_cmcrK = (1/(cnts**2)) * sum_z_cmcrK 

        # K x N
        XnK_distances = knn - normed_sum_z_cK + normed_sum_z_cmcrK
        
        # N x 1 
        assignments = XnK_distances.argmin(axis=0)

        assignment_matrix = create_assignment_matrix(K, assignments)
        # mu_k Can't be computed reliably because we don't know xn -> φ(xn)
        new_mus = compute_mus(K, X, assignment_matrix)
        mu_k[~np.isnan(new_mus)] = new_mus[~np.isnan(new_mus)]
        least_squares = compute_euc_distance(X, mu_k)
        losses[i] = (least_squares * assignment_matrix.squeeze()).sum()


    return mu_k, assignments, losses


centroids_mah, assigments_mah, losses_mah = kmeans_kernelised_distance(
    3, X, kernel_function=gaussian_kernel())

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