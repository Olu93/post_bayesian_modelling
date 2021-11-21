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
from viz import plot_train_val_curve, plot_w_samples
from sklearn.model_selection import train_test_split
from collections import Counter
# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True

# %%
n = 10000
w1_mu, w2_mu = 1, -3
w_cov = np.array([[1, -0.5], [-0.5, 1]])
w_mu = np.array([w1_mu, w2_mu])
w_distribution = stats.multivariate_normal(w_mu, w_cov)
true_w_sample = w_distribution.rvs()
n_w, n_c = 2, 3
xstd = 1000
val_n = 100
p_ord = 1
iterations = 20
smooth = 1
noise = True
seed = 42
true_X, true_y, W_c, P_c = observed_data_classification(
    n, xstd, n_w, n_c, noise, seed)

print(f"True weights")
print(W_c)

train_X, val_X, train_y, val_y = train_test_split(true_X,
                                                  true_y,
                                                  test_size=0.2)


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


display(compute_likelihood_distribution_gaussian(train_X, train_y))

# %%
import scipy

prior_per_class = compute_prior_classsize(train_X, train_y)
llh_per_class = compute_likelihood_distribution_gaussian(train_X, train_y)


def compute_posterior_probability(x_new, X, t, likelihood, prior):
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


compute_posterior_probability(train_X[0, None], train_X, train_y,
                              llh_per_class, prior_per_class)

# %%
correct = []
for x_new, y_true in zip(val_X, val_y):
    probs = compute_posterior_probability(x_new, train_X, train_y,
                                          llh_per_class, prior_per_class)
    correct.append(probs)
probs_df = pd.DataFrame(correct)
probs_df
# %%
from sklearn import metrics

metrics.accuracy_score(val_y, np.argmax(probs_df.values, axis=1))


# %%

# def compute_posterior_probability(X_new, X, t, likelihood, prior):
#     C = prior.keys()
#     all_joint_probabilities_per_class = {}
#     p_prior = {}
#     p_likelihood = {}
#     p_joint = {}
#     for c in C:
#         # P (Tnew = cjX; t)
#         p_prior[c] = prior[c]
#         # p(xnewjTnew = c; µc; Σc)
#         p_likelihood[c] = likelihood[c].pdf(x_new)
#         # p(xnewjTnew = c; µc; Σc)P (Tnew = cjX; t)
#         p_joint[c] = p_likelihood[c] * p_prior[c]
#         all_joint_probabilities_per_class[c] = p_joint[c]

#     p_data = sum(list(all_joint_probabilities_per_class.values()))
#     p_mu_posterior = {c: p_joint[c] / p_data for c in C}

#     return p_mu_posterior

def bayes_classifier(X_new, X, t, likelihood, prior):
    c_priors = prior(X, t)
    c_llhs = likelihood(X, t)
    single_predictions = []
    for x_new in X_new:
        probs = compute_posterior_probability(x_new, X, t, c_llhs, c_priors)
        single_predictions.append(probs)
    c_p_predictions = pd.DataFrame(single_predictions)
    
    pred_t = np.argmax(c_p_predictions.values, axis=1)
    return pred_t, probs_df


pred_y,  probs_df = bayes_classifier(val_X, train_X, train_y,
                            compute_likelihood_distribution_gaussian,
                            compute_prior_classsize)
metrics.accuracy_score(val_y, pred_y)
