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

train_X, true_y, X_means, X_covs = observed_data_classification_two_features(
    n, mean_data, cov_data)

# %%

ax = plt.gca()
plot_countours_and_points(ax, train_X, true_y, X_means, X_covs)
plt.show()


# %%
# Follows: Rogers, Simon, and Mark Girolami. A First Course in Machine Learning, Second Edition. 2nd ed. Chapman & Hall/CRC, 2016. https://nl1lib.org/book/11571143/9a0735.
def create_assignment_matrix(K, N, assignments):
    assignment_matrix = np.zeros((N, K, 1))
    assignment_matrix[range(N), assignments] = 1
    return assignment_matrix


# π_k: Average of probabilities of all X belonging to a posterior for a class
def update_pis(N, q_nk):
    # Start
    # q_nk  : K x N

    # K x 1
    sum_over_N = q_nk.sum(axis=-1)[:, None]
    # K x 1
    pi_k = sum_over_N / N
    return pi_k


# µ_k: Expected X value given the posterior for a class
def update_mus(X, q_nk):
    # Start
    # q_nk  : K x N
    # X     : N x F

    # K x N x 1
    q_nk1 = q_nk[:, :, None]

    # K x N x F
    q_nk1X = q_nk1 * X
    # K x F
    sum_q_kX = q_nk1X.sum(axis=1)
    # K x F = K x F / K x 1
    mus = sum_q_kX / q_nk.sum(axis=-1)[:, None]
    return mus


# Σ_k: Expected covariance value given the posterior for a class
def update_covs(X, mus, q_nk):
    # Start
    # q_nk  : K x N
    # mus   : K x F
    # X     : N x F

    N = len(X)

    # K x N x 1
    tmp = q_nk[:, :, None]

    # N x 1 x F
    new_X = X[:, None, :]
    # N x K x F
    diffs = new_X - mus
    # K x N x F
    k_diffs = np.transpose(diffs, axes=(1, 0, 2))
    # K x F x N
    k_diffs_2 = np.transpose(diffs, axes=(1, 2, 0))
    # K x F x F = K x F x N @ (K x N x F = K x N x 1 * K x N x F)
    # The dot product already sums all instances
    sq_diffs = k_diffs_2 @ (tmp * k_diffs)
    # K x F x F
    weighted_sum_features = (sq_diffs)
    # K x 1 x 1
    q_k = q_nk.sum(axis=-1)[:, None, None]
    # K x F x F
    covs = weighted_sum_features / q_k
    return covs

def update_covs_independence(X, mus, q_nk):
    # Start
    # q_nk  : K x N
    # mus   : K x F
    # X     : N x F

    N = len(X)

    # K x N x 1
    tmp = q_nk[:, :, None]

    # N x 1 x F
    new_X = X[:, None, :]
    # N x K x F
    diffs = new_X - mus
    # K x N x F
    k_diffs = np.transpose(diffs, axes=(1, 0, 2))

    # K x N x F = K x N x 1 * K x N x F
    sq_diffs = tmp * (k_diffs ** 2)

    # K x F x 1
    weigted_sum_sq_diffs = sq_diffs.sum(axis=1)[:, None]

    # 1 x F x F
    independence_covs = np.eye(X.shape[1])[None, :, :] 

    # K x 1 x 1
    q_k = q_nk.sum(axis=-1)[:, None, None]
    
    # K x F x F = (K x F x 1) @ (1 x F x F) / (K x 1 x 1)
    covs = (weigted_sum_sq_diffs * independence_covs) / q_k
    return covs

def update_covs_isotropic(X, mus, q_nk):
    # Start
    # q_nk  : K x N
    # mus   : K x F
    # X     : N x F

    N = len(X)

    # K x N x 1
    tmp = q_nk[:, :, None]

    # N x 1 x F
    new_X = X[:, None, :]
    # N x K x F
    diffs = new_X - mus
    # K x N x F
    k_diffs = np.transpose(diffs, axes=(1, 0, 2))

    # K x 1 x 1
    iso_var = np.var((k_diffs ** 2).unsqueeze(), axis=-1)[:, None, None]

    # K x N x 1 = K x N x 1 * K x 1 x 1
    sq_diffs = tmp * iso_var

    # K x 1 x 1
    weigted_sum_sq_diffs = sq_diffs.sum(axis=1)[:, None]

    # 1 x F x F
    independence_covs = np.eye(X.shape[1])[None, :, :] 

    # K x 1 x 1
    q_k = q_nk.sum(axis=-1)[:, None, None]
    
    # K x F x F = (K x 1 x 1) @ (1 x F x F) / (K x 1 x 1)
    covs = (weigted_sum_sq_diffs * independence_covs) / q_k
    return covs



# q_nk: Posterior of x being assigned to a class
def compute_posteriors(X, pis, mus, covs):
    all_posteriors = {}
    for k, (pi, mu, cov) in enumerate(zip(pis, mus, covs)):
        try:
            all_posteriors[k] = {
                'posterior': stats.multivariate_normal(mu, cov),
                'pi': pi
            }
        except Exception as e:
            print(e)

    return all_posteriors


def expectation(X, posteriors):
    joints = [v["pi"] * v["posterior"].pdf(X) for k, v in posteriors.items()]
    # K x N
    p_qnk_numerator = np.array(joints)
    # 1 x N
    p_qnk_denominator = p_qnk_numerator.sum(axis=0)[None, :]
    # K x N
    p_qnk = p_qnk_numerator / p_qnk_denominator
    return p_qnk


def maximization(N, X, q_nk, update_covs=update_covs):
    pis = update_pis(N, q_nk)
    mus = update_mus(X, q_nk)
    covs = update_covs(X, mus, q_nk)
    posteriors = compute_posteriors(X, pis, mus, covs)
    return posteriors


def em_algorithm(K, X, num_iter=10, restriction_level=0):
    update_covs_func = update_covs
    if restriction_level == 1:
        update_covs_func = update_covs_independence
    if restriction_level == 2:
        update_covs_func = update_covs_isotropic
    
    N, F = X.shape
    losses = np.zeros(num_iter)

    # Initialisation
    q_nk = np.ones((K, N)) / K
    assignments = np.random.randint(0, K, size=len(X))
    assignment_matrix = create_assignment_matrix(K, N, assignments)
    X_K = np.repeat(X[:, None, :], K, axis=1)
    mus = (assignment_matrix * X_K).sum(axis=0) / assignment_matrix.sum(axis=0)
    pis = update_pis(N, q_nk)
    covs = update_covs_func(X, mus, q_nk)
    posteriors = compute_posteriors(X, pis, mus, covs)
    # Run EM
    for i in range(num_iter):
        q_nk = expectation(X, posteriors)
        posteriors = maximization(N, X, q_nk, update_covs=update_covs_func)

    return posteriors, q_nk, losses


posteriors, q_nk, losses = em_algorithm(3, train_X, num_iter=100)
hard_assignments = q_nk.T.argmax(axis=1)
hard_assignments
# %%
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()

X = np.linspace(-5, 10, 1000)
Y = np.linspace(-5, 10, 1000)
X, Y = np.meshgrid(X, Y)
flat_data = np.array([X.flatten(), Y.flatten()]).T

for c, class_items in posteriors.items():
    Z = class_items['posterior'].pdf(flat_data).reshape(X.shape)
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)

    members = (hard_assignments == c).flatten()
    if np.any(members):
        X_likelihoods = q_nk.T[members]
        X_members = train_X[members]
        ax.scatter(X_members[:, 0], X_members[:, 1])
plt.show()

# %%

