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
from classification_bayes_classifier import compute_likelihood_distribution_gaussian, compute_prior_classsize, bayes_classifier, compute_prior_uniform
from data import observed_data_classification, observed_data_classification_two_features
from helper import add_bias_vector, compute_metrics, create_polinomial_bases, predict, sigmoid
from viz import plot_countours_and_points, plot_train_val_curve, plot_w_samples
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics

# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True

# %%
n = 1000
n_c = 3
x1mu, x2mu = 1, 1
x1std, x2std = 3, 3
xystd = 2.5
val_n = 100
noise = True
seed = 42
cov_data = np.array([[x1std, xystd], [xystd, x2std]])
mean_data = np.array([x1mu, x2mu])

true_X, true_y, X_means, X_covs = observed_data_classification_two_features(
    n, mean_data, cov_data)

# %%
train_X, val_X, train_y, val_y = train_test_split(true_X,
                                                  true_y,
                                                  test_size=0.2)

ax = plt.gca()
plot_countours_and_points(ax, train_X, train_y, X_means, X_covs)
plt.show()


# %%
# Distributions for each class - p(x_new|T_new = c; X; t)
#   - Assumes that classes are independent from another sum(P(T_new=c|T_other=c)) = P(T_new=c)
#   - Assumes every feature is independent from each other
def compute_likelihood_distribution_gaussian_naive(X, t):
    class_counts = Counter(np.sort(t.flat))
    all_class_llh = {}
    c_params = {}
    for c, cnt_c in class_counts.items():
        t_subset_idxs = (t == c).flat
        sum_of_X = X[t_subset_idxs].sum(axis=0)[None, :]
        mu_c = sum_of_X / cnt_c
        difference = X[t_subset_idxs] - mu_c
        sq_diff = difference**2
        mean_sq_diff = sq_diff.mean(axis=0)
        c_params[c] = {"mu": mu_c, "var": mean_sq_diff}

    for c, params in c_params.items():
        all_class_llh[c] = stats.multivariate_normal(
            params['mu'].flat, params['var'] @ np.eye(len(params['mu'].flat)))

    return all_class_llh


all_llhs = compute_likelihood_distribution_gaussian_naive(train_X, train_y)
all_llhs


# %%
def compute_joint_probability(X_new, likelihood, prior):
    # P (T_new = cjX; t)
    p_prior = prior
    # p(X_new|T_new = c; µc; Σc)
    p_likelihood = likelihood.pdf(X_new)
    p_joint = p_likelihood * p_prior
    return p_joint


def naive_bayes_classifier(X_new, X, t, likelihood, prior):
    c_priors = prior(X, t)
    C = c_priors.keys()
    c_llhs = likelihood(X, t)
    unnormalised_posterior = {}
    for c in C:
        c_prior = c_priors[c]
        c_llh = c_llhs[c]
        unnormalised_joint_probability = compute_joint_probability(
            X_new, c_llh, c_prior)
        unnormalised_posterior[c] = unnormalised_joint_probability
    df_unnormalised_posterior = pd.DataFrame(unnormalised_posterior)
    # p(xnew|Tnew = c; µc; Σc)P (Tnew = c|X; t)
    df_norm_constants = np.sum(df_unnormalised_posterior.values, axis=1)
    normalized_posterior_probs = df_unnormalised_posterior.values / df_norm_constants[:,
                                                                                      None]
    return normalized_posterior_probs


c_p_predictions = naive_bayes_classifier(
    val_X, train_X, train_y, compute_likelihood_distribution_gaussian_naive,
    compute_prior_uniform)
pred_y = np.argmax(c_p_predictions, axis=1) + 1
metrics.accuracy_score(val_y, pred_y)

# %%
c_p_predictions = bayes_classifier(val_X, train_X, train_y,
                                   compute_likelihood_distribution_gaussian,
                                   compute_prior_uniform)
pred_y = np.argmax(c_p_predictions, axis=1) + 1
metrics.accuracy_score(val_y, pred_y)

# %%
