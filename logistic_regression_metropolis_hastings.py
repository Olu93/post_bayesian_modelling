# %%
import abc
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import meshgrid
from numpy.random import multivariate_normal
import pandas as pd
from IPython.display import display
from scipy import stats
from scipy import special
from matplotlib import cm
import random as r
from data import observed_data, observed_data_binary, observed_data_linear, true_function_polynomial, true_function_sigmoid
from helper import add_bias_vector, create_polinomial_bases, log_stable, sigmoid
from tqdm.notebook import tqdm
from matplotlib.patches import Ellipse

from viz import plot_w_path_until_burnin, plot_w_samples

# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True
# np.random.seed(42)
# %%

n = 10000
w1_mu, w2_mu = 1, -3
w_cov = np.array([[1, -0.5], [-0.5, 1]])
w_mu = np.array([w1_mu, w2_mu])
w_distribution = stats.multivariate_normal(w_mu, w_cov)
true_w_sample = w_distribution.rvs()
w1, w2 = true_w_sample[0], true_w_sample[1]
xstd = 1000
val_n = 100
p_ord = 1
iterations = 50
smooth = 1
noise = 0
data = np.vstack(np.array(observed_data_binary(n, w1, w2, xstd, noise)).T)
val_data = np.vstack(np.array(observed_data_binary(val_n, w1, w2, xstd, noise)).T)
print(f"True weights are : {w1, w2}")

train_X = data[:, :-1]
# train_X = add_bias_vector(create_polinomial_bases(data[:, :-1], p_ord))
train_y = data[:, -1][:, None]
val_X = val_data[:, :-1]
# val_X = add_bias_vector(create_polinomial_bases(val_data[:, :-1], p_ord))
val_y = val_data[:, -1][:, None]

## %%
# Prior p(w|σ²) = N (0; σ²I) | w ~ N (0; σ²I)
# Likelihood p(t|X; w) = Π p(T_n = t_n|x_n; w)**t_n * p(T_n = t_n|x_n; w)**(1-t_n)
# Posterior: p(w|X, t, σ²)
# Predictive Distribution: P(T_new = 1|x_new; X; t; σ²) ≃ (1/N_s) Σ P(T_new = 1|x_new; w_s)
#   - Predictive distribution is asymptotically equal to the average draws with a sampled posterior value w_s


def metropolis_hastings_algorithm_diagonal(X, t, w_init, w_cov_prior, sigma_sq=None, num_iter=1000):
    num_features = len(w_init)
    all_ws = np.zeros((num_iter + 1, num_features, 1))
    w_last = np.random.multivariate_normal(w_init.flat, w_cov_prior)[:, None]
    w_current = w_last
    all_ws[0] = w_last
    pbar = tqdm(range(1, num_iter))
    for i in pbar:
        w_candidate = propose_new_sample(w_last, w_cov_prior)
        is_accepted = accept_or_reject_sample(w_candidate, w_current, X, t, w_init, w_cov_prior)
        w_selected = w_candidate if is_accepted else w_current
        w_current, w_selected, w_last = update_w_params(i, all_ws, w_current, w_selected)
        m_train_loss, m_train_acc = compute_metrics(w_current, train_X, train_y)
        pbar.set_description_str(f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return w_current, all_ws.reshape(num_iter + 1, -1)


def metropolis_hastings_algorithm_orthogonal(X, t, w_init, w_cov_prior, sigma_sq=None, num_iter=1000):
    num_features = len(w_init)
    all_ws = np.zeros((num_iter + 1, num_features, 1))
    w_last = np.random.multivariate_normal(w_init.flat, w_cov_prior)[:, None]
    w_current = w_last
    all_ws[0] = w_last
    pbar = tqdm(range(1, num_iter))
    for i in pbar:
        for j in range(num_features):
            dim_manipulation_cov_prior = np.zeros_like(w_cov_prior)
            dim_manipulation_cov_prior[j, j] = w_cov_prior[j, j]
            w_candidate = propose_new_sample(w_last, dim_manipulation_cov_prior)
            is_accepted = accept_or_reject_sample(w_candidate, w_current, X, t, w_init, w_cov_prior)
            w_selected = w_candidate if is_accepted else w_current
            w_current, w_selected, w_last = update_w_params(i, all_ws, w_current, w_selected)
            m_train_loss, m_train_acc = compute_metrics(w_current, train_X, train_y)
            pbar.set_description_str(f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return w_selected, all_ws.reshape(num_iter + 1, -1)


def update_w_params(i, all_ws, w_current, w_selected):
    w_current = w_selected
    all_ws[i] = w_selected
    w_last = all_ws[i - 1]
    return w_current, w_selected, w_last


# p(w_candidate|w_last_accepted, Σ) = N(w_last_accepted; Σ)
def propose_new_sample(w_last_accepted, w_cov):
    w_candidate = np.random.multivariate_normal(w_last_accepted.flat, w_cov)[:, None]
    return w_candidate


def accept_or_reject_sample(w_candidate, w_last, X, t, w_mu, w_cov):
    thresh = 1.0
    inv_sampling_val = np.random.uniform()
    log_prior_candidate = stats.multivariate_normal.logpdf(w_candidate.T, w_mu.flat, w_cov)
    log_prior_last = stats.multivariate_normal.logpdf(w_last.T, w_mu.flat, w_cov)
    log_likelihood_candidate = log_likelihood_function(w_candidate, X, t)
    log_likelihood_last = log_likelihood_function(w_last, X, t)
    log_r_candidate = log_prior_candidate + log_likelihood_candidate
    log_r_last = log_prior_last + log_likelihood_last
    r = log_r_candidate - log_r_last
    # r = np.exp(r)

    # r = np.log(r)
    thresh = np.log(thresh)
    inv_sampling_val = np.log(inv_sampling_val)

    is_accepted = True if (r >= thresh) else (inv_sampling_val < r)

    str_text = f"{'Y' if is_accepted else 'F'}: "
    if (r >= thresh):
        str_text = str_text + f'r={r:.4f} >= t={thresh:.4f} | automatic accept'
    else:
        if (inv_sampling_val < r):
            str_text = str_text + f'u={inv_sampling_val:4.4f} < r={r:.4f}'
        if not (inv_sampling_val < r):
            str_text = str_text + f'u={inv_sampling_val:4.4f} < r={r:.4f}'
    # print(f"{str_text}, {w_candidate.T}")

    return is_accepted


def log_likelihood_function(w, X, t):
    probabilities = sigmoid(X @ w)
    ones = log_stable(probabilities)
    zeros = log_stable((1 - probabilities))
    all_log_likelihoods = t * ones + (1 - t) * zeros
    return np.sum(all_log_likelihoods)


def compute_metrics(w, X, y):
    y_hat = sigmoid(X @ w)
    losses = y - y_hat
    m_loss = np.mean(np.abs(losses))
    m_acc = np.mean(y == ((y_hat >= 0.5) * 1.0))
    return m_loss, m_acc


def select_matrix_cross(index, square_matrix):
    masked_matrix = np.zeros_like(square_matrix)
    selection_matrix = np.full_like(square_matrix, False, bool)
    selection_matrix[:, index] = True
    selection_matrix[index, :] = True
    masked_matrix[selection_matrix] = square_matrix[selection_matrix]
    return masked_matrix


# w_hat = np.random.normal(1, 0.5, size=(train_X.shape[1], 1)) * 2
# w = np.zeros(shape=(train_X.shape[1], 1))
# w = np.array([[1], [-3], [3]])
# w = np.random.multivariate_normal([w1,w2], np.eye(train_X.shape[1]))[:, None]
# w = np.random.multivariate_normal([1, w1, w2], np.eye(train_X.shape[1]))[:, None]

all_deltas = []
all_train_accs = []
all_val_accs = []
assumed_sigma_sq = 1
w_mu_prior = np.ones((train_X.shape[1], 1))
w_cov_prior = np.eye(train_X.shape[1]) * assumed_sigma_sq
num_iter = 1000
# print("====================")

w_hat_diagonal, all_w_hats_diagonal = metropolis_hastings_algorithm_diagonal(train_X,
                                                                             train_y,
                                                                             w_mu_prior,
                                                                             w_cov_prior,
                                                                             assumed_sigma_sq,
                                                                             num_iter=num_iter)
w_hat_orthogonal, all_w_hats_orthogonal = metropolis_hastings_algorithm_orthogonal(train_X,
                                                                                   train_y,
                                                                                   w_mu_prior,
                                                                                   w_cov_prior,
                                                                                   assumed_sigma_sq,
                                                                                   num_iter=num_iter)

print("Diag: Expected Mean W", all_w_hats_diagonal[num_iter // 10:].mean(axis=0))
print("Orth: Expected Mean W", all_w_hats_orthogonal[num_iter // 10:].mean(axis=0))

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 18))
burn_in_period = num_iter // 1

plot_w_path_until_burnin(all_w_hats_diagonal, ax1, w_cov, w_mu, burn_in_period, title="Diagonal", precision=2)
plot_w_path_until_burnin(all_w_hats_orthogonal, ax2, w_cov, w_mu, burn_in_period, title="Orthogonal", precision=2)
fig.tight_layout()
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 18))
burn_in_period = num_iter // 100
plot_w_samples(all_w_hats_diagonal, ax1, w_cov, w_mu, burn_in_period, title="Diagonal", precision=2)
plot_w_samples(all_w_hats_orthogonal, ax2, w_cov, w_mu, burn_in_period, title="Orthogonal", precision=2)
fig.tight_layout()
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

ax1.plot(all_w_hats_diagonal)
ax2.plot(all_w_hats_orthogonal)
fig.tight_layout()
plt.show()