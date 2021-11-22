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
xystd = 1
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
# P(T_new = c|X; t)
# Uniform prior: Just assign equal probability to each class
def compute_prior_uniform(X, t):
    class_counts = Counter(np.sort(t.flat))
    num_classes = max(list(set(class_counts.keys()))) + 1
    return {c: 1 / num_classes for c, _ in class_counts.items()}


# P(T_new = c|X; t)
# Class size prior: N_c/N
def compute_prior_classsize(X, t):
    class_counts = Counter(np.sort(t.flat))
    N = len(t)
    return {c: N_c / N for c, N_c in class_counts.items()}


display(compute_prior_uniform(train_X, train_y))
display(compute_prior_classsize(train_X, train_y))


# %%
# Distributions for each class - p(x_new|T_new = c; X; t)
#   - Assumes that classes are independent from another sum(P(T_new=c|T_other=c)) = P(T_new=c)
def compute_likelihood_distribution_gaussian(X, t):
    class_counts = Counter(np.sort(t.flat))
    all_class_llh = {}
    for c, cnt_c in class_counts.items():
        t_subset_idxs = (t == c).flat
        sum_of_X = X[t_subset_idxs].sum(axis=0)[None, :]
        mu_c = sum_of_X / cnt_c
        difference = X[t_subset_idxs] - mu_c
        sigma_c = difference.T @ difference / cnt_c
        all_class_llh[c] = stats.multivariate_normal(mu_c.flat, sigma_c)

    return all_class_llh


all_llhs = compute_likelihood_distribution_gaussian(train_X, train_y)

for i in [1, 2, 3]:
    print(f"========= Class: {i} =========")
    print(f"--- Mean: {i} ---")
    display(X_means[i - 1])
    display(all_llhs[i].mean)
    print(f"--- Cov: {i} ---")
    display(X_covs[i - 1])
    display(all_llhs[i].cov)

# %%
prior_per_class = compute_prior_classsize(train_X, train_y)
llh_per_class = compute_likelihood_distribution_gaussian(train_X, train_y)
for i in [1, 2, 3]:
    print(f"========= Likelihoods: {i} =========")
    print(f"--- Mean: {i} ---")
    display(llh_per_class[i].mean)
    print(f"--- Cov: {i} ---")
    display(llh_per_class[i].cov)


# %%
def compute_posterior_probability_slow(x_new, X, t, likelihood, prior):
    C = prior.keys()
    all_joint_probabilities_per_class = {}
    p_prior = {}
    p_likelihood = {}
    p_joint = {}
    for c in C:
        # P (Tnew = cjX; t)
        p_prior[c] = prior[c]
        # p(xnewjTnew = c; µc; Σc)
        p_likelihood[c] = likelihood[c].pdf(x_new)
        # p(xnewjTnew = c; µc; Σc)P (Tnew = cjX; t)
        p_joint[c] = p_likelihood[c] * p_prior[c]
        all_joint_probabilities_per_class[c] = p_joint[c]

    p_data = sum(list(all_joint_probabilities_per_class.values()))
    p_mu_posterior = {c: p_joint[c] / p_data for c in C}

    return p_mu_posterior


compute_posterior_probability_slow(val_X[1, None], train_X, train_y,
                                   llh_per_class, prior_per_class)

# %%
correct = []
for x_new, y_true in zip(val_X, val_y):
    probs = compute_posterior_probability_slow(x_new, train_X, train_y,
                                               llh_per_class, prior_per_class)
    correct.append(probs)
probs_df = pd.DataFrame(correct)
probs_df
# %%

metrics.accuracy_score(val_y, probs_df.idxmax(axis=1))


# %%
def bayes_classifier_slow(X_new, X, t, likelihood, prior):
    c_priors = prior(X, t)
    c_llhs = likelihood(X, t)
    single_predictions = []
    for x_new in X_new:
        probs = compute_posterior_probability_slow(x_new, X, t, c_llhs,
                                                   c_priors)
        single_predictions.append(probs)
    c_p_predictions = pd.DataFrame(single_predictions)

    pred_t = c_p_predictions.idxmax(axis=1)
    return pred_t, probs_df


pred_y, probs_df = bayes_classifier_slow(
    val_X, train_X, train_y, compute_likelihood_distribution_gaussian,
    compute_prior_classsize)
metrics.accuracy_score(val_y, pred_y)


# %%
def compute_joint_probability(X_new, likelihood, prior):
    # P (T_new = cjX; t)
    p_prior = prior
    # p(X_new|T_new = c; µc; Σc)
    p_likelihood = likelihood.pdf(X_new)
    p_joint = p_likelihood * p_prior
    return p_joint


def bayes_classifier(X_new, X, t, likelihood, prior):
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


c_p_predictions = bayes_classifier(val_X, train_X, train_y,
                                   compute_likelihood_distribution_gaussian,
                                   compute_prior_classsize)

pred_t = np.argmax(c_p_predictions, axis=1) + 1
metrics.accuracy_score(val_y, pred_y)

# %%
