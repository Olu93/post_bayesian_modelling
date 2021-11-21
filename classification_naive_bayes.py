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
from viz import plot_train_val_curve, plot_w_samples
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics

# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True

# %%
n = 10000
n_c = 3
xystd = -3
x1mu, x2mu = 1,3
x1std, x2std = 1, 1
val_n = 100
noise = True
seed = 42
cov_data = np.array([[x1std, xystd], [xystd, x2std]])
mean_data = np.array([x1mu, x2mu])

true_X, true_y, W_c, P_c = observed_data_classification_two_features(
    n, mean_data, cov_data, n_c, noise, seed)

print(f"True weights")
print(W_c)

train_X, val_X, train_y, val_y = train_test_split(true_X,
                                                  true_y,
                                                  test_size=0.2)


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


display(compute_likelihood_distribution_gaussian_naive(train_X, train_y))


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
pred_y = np.argmax(c_p_predictions, axis=1)
metrics.accuracy_score(val_y, pred_y)

# %%
c_p_predictions = bayes_classifier(val_X, train_X, train_y,
                                   compute_likelihood_distribution_gaussian,
                                   compute_prior_uniform)
pred_y = np.argmax(c_p_predictions, axis=1)
metrics.accuracy_score(val_y, pred_y)

# %%
def compute_joint_probability(X_new, likelihood, prior):
    # P (T_new = cjX; t)
    p_prior = prior
    # p(X_new|T_new = c; µc; Σc)
    p_likelihood = likelihood.pdf(X_new)
    p_joint = p_likelihood * p_prior
    return p_joint


def naive_bayes_classifier(X_new, X, t):
    class_counts = Counter(np.sort(t.flat))
     = {}
    for c, c_cnt in class_counts.items():

        t_subset_idxs = (t == c).flat
        #  n = instances, m = features
        sum_x_n = X[t_subset_idxs].sum(axis=0)[None, :]
        sum_x_nm = sum_x_n.sum()
        q_cm = sum_x_n / sum_x_nm
        


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
pred_y = np.argmax(c_p_predictions, axis=1)
metrics.accuracy_score(val_y, pred_y)
