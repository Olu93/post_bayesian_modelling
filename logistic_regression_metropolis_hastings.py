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
from data import observed_data, observed_data_binary, observed_data_linear, true_function_polynomial, true_function_sigmoid
from helper import add_bias_vector, create_polinomial_bases
from tqdm.notebook import tqdm

# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True

# %%


def true_function_sigmoid(x, y, w1, w2):
    return 1 / (1 + np.exp(w1 * x + w2 * y))


def observed_data_binary(d: int = 10, w1=2, w2=2, std=3, with_err=False):
    data = np.random.normal(0, std, size=(d, 2))
    # data = np.random.uniform(-std, std, size=(d, 2))
    # print(data)
    x, y = data[:, 0], data[:, 1]
    probability = true_function_sigmoid(x, y, w1, w2)
    if with_err:
        err = np.random.randn(d) * 0.10
        probability = probability + err
    z = probability < 0.5
    return x, y, z


n = 1000
w1, w2 = -6, 1
xstd = 1
val_n = 100
p_ord = 1
iterations = 50
smooth = 1
data = np.vstack(np.array(observed_data_binary(n, w1, w2, xstd, True)).T)
val_data = np.vstack(np.array(observed_data_binary(val_n, w1, w2, xstd, True)).T)
display(data.max(axis=0))
display(data.min(axis=0))
display(data.mean(axis=0))

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


# %%
def metropolis_hastings_algorithm_diagonal(X, t, w_init, w_cov_prior, sigma_sq=None, num_iter=1000):
    all_ws = np.zeros((num_iter, len(w_init)))
    w_init_flat = w_init.flatten()
    w = np.random.multivariate_normal(w_init_flat, w_cov_prior)
    pbar = tqdm(range(num_iter - 1))
    all_ws[0] = w
    w_last = all_ws[0]
    w_current = all_ws[0]
    for i in pbar:
        w_candidate = propose_new_sample(w_last, w_cov_prior)
        is_accepted = accept_or_reject_sample(w_candidate, w_current, X, t, w_init_flat, w_cov_prior)
        w = w_candidate if is_accepted else w
        all_ws[i + 1] = np.copy(w)
        w_last = all_ws[i]
        w_current = all_ws[i + 1]
        train_preds = train_X @ w
        train_losses = train_y - train_preds
        m_train_loss = np.mean(np.abs(train_losses))
        m_train_acc = np.mean(train_y == (train_preds > 0.5) * 1.0)
        pbar.set_description_str(f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return w, all_ws


def metropolis_hastings_algorithm_perpendicular(X, t, w_init, w_cov_prior, sigma_sq=None, num_iter=1000):
    all_ws = np.zeros((num_iter, len(w_init)))
    w_init_flat = w_init.flatten()
    w = np.random.multivariate_normal(w_init_flat, w_cov_prior)
    pbar = tqdm(range(num_iter - 1))
    all_ws[0] = w
    w_last = all_ws[0]
    w_current = all_ws[0]
    for i in pbar:
        for j in range(len(w_init_flat)):
            dim_manipulation_cov_prior = np.zeros_like(w_cov_prior)
            dim_manipulation_cov_prior[j, j] = w_cov_prior[j, j]
            # dim_manipulation_cov_prior = select_matrix_cross(j, w_cov_prior)
            w_candidate = propose_new_sample(w_last, dim_manipulation_cov_prior)
            is_accepted = accept_or_reject_sample(w_candidate, w_current, X, t, w_init_flat, w_cov_prior)
            w = w_candidate if is_accepted else w
            all_ws[i + 1] = np.copy(w)
            w_last = all_ws[i]
            w_current = all_ws[i + 1]
            train_preds = train_X @ w
            train_losses = train_y - train_preds
            m_train_loss = np.mean(np.abs(train_losses))
            m_train_acc = np.mean(train_y == (train_preds > 0.5) * 1.0)
            pbar.set_description_str(f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return w, all_ws


def select_matrix_cross(index, square_matrix):
    masked_matrix = np.zeros_like(square_matrix)
    selection_matrix = np.full_like(square_matrix, False, bool)
    selection_matrix[:, index] = True
    selection_matrix[index, :] = True
    masked_matrix[selection_matrix] = square_matrix[selection_matrix]
    return masked_matrix


# p(w_candidate|w_last_accepted, Σ) = N(w_last_accepted; Σ)
def propose_new_sample(w_last_accepted, w_cov):
    w_candidate = np.random.multivariate_normal(w_last_accepted, w_cov)
    # w_candidate = np.copy(w_last_accepted)
    # w_candidate[w_index] = w_candidate[w_index] + w_delta[w_index]
    return w_candidate


def accept_or_reject_sample(w_candidate, w_last, X, t, w_mu, w_cov):
    thresh = 1
    log_prior_candidate = stats.multivariate_normal.logpdf(w_candidate, w_mu, w_cov)
    log_prior_last = stats.multivariate_normal.logpdf(w_last, w_mu, w_cov)
    log_likelihood_candidate = log_likelihood_function(w_candidate, X, t)
    log_likelihood_last = log_likelihood_function(w_last, X, t)
    r_candidate = log_prior_candidate + log_likelihood_candidate
    r_last = log_prior_last + log_likelihood_last
    r = np.exp(r_candidate - r_last)
    inv_sampling_val = np.random.uniform()

    # r = np.log(r)
    # thresh = np.log(thresh)
    # inv_sampling_val = np.log(inv_sampling_val)

    is_accepted = True if (r > thresh) else (inv_sampling_val < r)
    # print(f"{is_accepted} : {inv_sampling_val:.4f} < {r:.4f}{'*' if r>thresh else ''}, {w_candidate.T}")
    return is_accepted


def log_likelihood_function(w, X, t):
    probabilities = sigmoid(X @ w)[:, None]
    all_log_likelihoods = (t * np.log(probabilities)) + (1 - t) * np.log((1 - probabilities))
    return np.nansum(all_log_likelihoods)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
num_iter = 200
# print("====================")

w_hat, all_w_hats = metropolis_hastings_algorithm_perpendicular(train_X,
                                                                train_y,
                                                                w_mu_prior,
                                                                w_cov_prior,
                                                                assumed_sigma_sq,
                                                                num_iter=num_iter)
print("Expected Mean W", all_w_hats[num_iter // 10:].mean(axis=0))


## %%
def predict(ws, val_X, num_samples=1000):
    logits = val_X @ ws.T
    probabilities = sigmoid(logits).mean(axis=1)
    return probabilities[:, None]


## %%
fig, (ax1) = plt.subplots(1, 1, figsize=(15, 15))
burnin_period = num_iter // 1

w_cov = np.array([[1, 0], [0, 1]])
w_mu = np.array([w1, w2])

X = np.linspace(w_mu[0] - w_cov[0, 0], w_mu[0] + w_cov[0, 0], 100)
Y = np.linspace(w_mu[1] - w_cov[1, 1], w_mu[1] + w_cov[1, 1], 100)
X, Y = np.meshgrid(X, Y)
distribution = stats.multivariate_normal(w_mu, w_cov)
Z_True = distribution.pdf(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

CS = ax1.contour(X, Y, Z_True)
# ax1.clabel(CS, inline=True, fontsize=10)
ax1.plot(all_w_hats[:burnin_period, -2], all_w_hats[:burnin_period, -1])
ax1.scatter(all_w_hats[0][0], all_w_hats[0][1], s=100, c='green', label="start")
ax1.scatter(all_w_hats[burnin_period - 1][-2], all_w_hats[burnin_period - 1][-1], s=100, c='red', label="end")
ax1.set_xlabel("w1")
ax1.set_ylabel("w2")
# ax1.legend()
ax1.set_title("True Distribution")

# %%

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
burnin_period = num_iter // 10

w_cov = np.array([[1, 0], [0, 1]])
w_mu = np.array([w1, w2])

X = np.linspace(w_mu[0] - w_cov[0, 0], w_mu[0] + w_cov[0, 0], 100)
Y = np.linspace(w_mu[1] - w_cov[1, 1], w_mu[1] + w_cov[1, 1], 100)
X, Y = np.meshgrid(X, Y)
distribution = stats.multivariate_normal(w_mu, w_cov)
Z_True = distribution.pdf(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

CS = ax1.contour(X, Y, Z_True)
# ax1.clabel(CS, inline=True, fontsize=10)
ax1.scatter(all_w_hats[burnin_period:, -1], all_w_hats[burnin_period:, -2])
ax1.set_xlabel("w1")
ax1.set_ylabel("w2")
# ax1.legend()
ax1.set_title("True Distribution")

X = np.linspace(w_mu_prior[-2] - 2 * np.sqrt(w_cov_prior)[-2, -2], w_mu_prior[-2] + 2 * np.sqrt(w_cov_prior)[-2, -2],
                100)
Y = np.linspace(w_mu_prior[-1] - 2 * np.sqrt(w_cov_prior)[-1, -1], w_mu_prior[-1] + 2 * np.sqrt(w_cov_prior)[-1, -1],
                100)
X, Y = np.meshgrid(X, Y)

pred_distribution = stats.multivariate_normal([0, 0], 1)
Z_Pred = pred_distribution.pdf(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

CS = ax2.contour(X, Y, Z_Pred)
# CS = ax1.contour(X, Y, Z_Pred)
ax2.scatter(all_w_hats[burnin_period:, -1], all_w_hats[burnin_period:, -2])
ax2.set_xlabel("w1")
ax2.set_ylabel("w2")
# ax2.legend()
ax2.set_title("Approximation close-up")
# ax2.set_zlim3d(ax1.get_xlim3d())
# ax2.set_zlim3d(ax1.get_ylim3d())
# ax2.set_zlim3d(ax1.get_zlim3d())
# fig.tight_layout()
plt.show()

# %%
plt.plot(all_w_hats)