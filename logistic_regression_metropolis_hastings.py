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
    return 1 / (1 + np.exp(-(w1 * x + w2 * y)))


def observed_data_binary(d: int = 10, w1=2, w2=2, std=3, with_err=False):
    # data = np.random.normal(0, std, size=(d, 2))
    data = np.random.uniform(-std, std, size=(d, 2))
    # print(data)
    x, y = data[:, 0], data[:, 1]
    probability = true_function_sigmoid(x, y, w1, w2)
    if with_err:
        err = np.random.randn(d) * 0.10
        probability = probability + err
    z = (probability >= 0.5) * 1
    return x, y, z


n = 1000
w1, w2 = 1, 3
xstd = 10
val_n = 100
p_ord = 1
iterations = 50
smooth = 1
data = np.vstack(np.array(observed_data_binary(n, w1, w2, xstd, False)).T)
val_data = np.vstack(np.array(observed_data_binary(val_n, w1, w2, xstd, False)).T)
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


## %%
def metropolis_hastings_algorithm_diagonal(X, t, w_init, w_cov_prior, sigma_sq=None, num_iter=1000):
    all_ws = np.zeros((num_iter, len(w_init), 1))
    num_features = len(w_init)
    w_last = w_init
    w_current = np.random.multivariate_normal(w_init.flat, w_cov_prior)
    pbar = tqdm(range(num_iter - 1))
    for i in pbar:
        w_candidate = propose_new_sample(w_last, w_cov_prior)
        is_accepted = accept_or_reject_sample(w_candidate, w_current, X, t, w_init, w_cov_prior)
        w_current = w_candidate if is_accepted else w_current
        all_ws[i + 1] = np.copy(w_current)
        w_last = all_ws[i]
        w_current = all_ws[i + 1]
        m_train_loss, m_train_acc = compute_metrics(w_current, train_X, train_y)
        pbar.set_description_str(f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return w_current, all_ws.reshape(num_iter, -1)


def metropolis_hastings_algorithm_perpendicular(X, t, w_init, w_cov_prior, sigma_sq=None, num_iter=1000):
    all_ws = np.zeros((num_iter, len(w_init), 1))
    num_features = len(w_init)
    w_last = w_init
    w_current = np.random.multivariate_normal(w_init.flat, w_cov_prior)[:, None]
    pbar = tqdm(range(num_iter - 1))
    for i in pbar:
        for j in range(num_features):
            dim_manipulation_cov_prior = np.zeros_like(w_cov_prior)
            dim_manipulation_cov_prior[j, j] = w_cov_prior[j, j]
            w_candidate = propose_new_sample(w_last, dim_manipulation_cov_prior)
            is_accepted = accept_or_reject_sample(w_candidate, w_current, X, t, w_init, w_cov_prior)
            w_selected = w_candidate if is_accepted else w_last
            all_ws[i + 1] = np.copy(w_selected)
            w_last = all_ws[i]
            w_current = all_ws[i + 1]
            m_train_loss, m_train_acc = compute_metrics(w_current, train_X, train_y)
            pbar.set_description_str(f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return w_selected, all_ws.reshape(num_iter, -1)


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
    r = np.exp(r)

    r = np.log(r)
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
    print(f"{str_text}, {w_candidate.T}")

    return is_accepted


def log_likelihood_function(w, X, t):
    probabilities = sigmoid(X @ w)
    all_log_likelihoods = (t * np.log(probabilities)) + (1 - t) * np.log((1 - probabilities))
    return np.nansum(all_log_likelihoods)


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def compute_metrics(w, X, y):
    train_preds = sigmoid(X @ w)
    train_losses = y - train_preds
    m_train_loss = np.mean(np.abs(train_losses))
    m_train_acc = np.mean(y == ((train_preds >= 0.5) * 1.0))
    return m_train_loss, m_train_acc


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
num_iter = 100
# print("====================")

w_hat, all_w_hats = metropolis_hastings_algorithm_diagonal(train_X,
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
ax1.scatter(all_w_hats[0][-2], all_w_hats[0][-1], s=100, c='green', label="start")
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