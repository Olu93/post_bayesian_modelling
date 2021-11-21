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
from data import observed_data, observed_data_binary, observed_data_linear, true_function_polynomial
from helper import add_bias_vector, compute_metrics, create_polinomial_bases, sigmoid
from tqdm.notebook import tqdm

from viz import plot_train_val_curve, plot_w_path_from_burnin
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
w1, w2 = true_w_sample[0], true_w_sample[1]
xstd = 1000
val_n = 100
p_ord = 1
iterations = 20
smooth = 1
noise = 0
data = np.vstack(np.array(observed_data_binary(n, w1, w2, xstd, noise)).T)
val_data = np.vstack(
    np.array(observed_data_binary(val_n, w1, w2, xstd, noise)).T)
print(f"True weights are : {w1, w2}")

train_X = data[:, :-1]
# train_X = add_bias_vector(create_polinomial_bases(data[:, :-1], p_ord))
train_y = data[:, -1][:, None]
val_X = val_data[:, :-1]
# val_X = add_bias_vector(create_polinomial_bases(val_data[:, :-1], p_ord))
val_y = val_data[:, -1][:, None]
# %%
# Prior p(w|σ²) = N (0; σ²I) | w ~ N (0; σ²I)
# Likelihood p(t|X; w) = Π p(T_n = t_n|x_n; w)**t_n * p(T_n = t_n|x_n; w)**(1-t_n)
#   - T_n is binary
#   - Probability of hit:   p(T_n = 1|x_n; w) = sigmoid(w.T | x_n)
#   - Counter probability   p(T_n = 0|x_n; w) = 1 - p(T_n = 1|x_n; w)
# Posterior: p(w|X, t, σ²)
#   - Is hard to compute due to the normalizing constant
#   - We can optimize the numerator instead g(w; X, t, σ²) = p(t|X; w) * p(w|σ²)
#   - Best is to optimize the log g(w; X, t, σ²) = p(t|X; w) + p(w|σ²)

def stochastic_gradient_descent(X, t, w_init, sigma_sq, lr, num_iter, first_derivation):

    num_features = X.shape[1]
    num_true_run_length = num_iter + 1
    all_ws = np.zeros((num_true_run_length, num_features, 1))
    all_deltas = np.zeros((num_true_run_length, num_features, 1))
    all_gradients = np.zeros((num_true_run_length, num_features, 1))
    gradient = first_derivation(w_init, X, t, sigma_sq)
    w = w_init + lr * 1/len(X) * gradient
    all_ws[0] = w

    pbar = tqdm(range(1, num_true_run_length))
    for i in pbar:
        gradient = first_derivation(w, X, t, sigma_sq)
        weight_change = lr * 1/len(X) * gradient
        w = w + weight_change
        all_ws[i] = w
        all_deltas[i] = weight_change
        all_gradients[i] = gradient
        m_train_loss, m_train_acc = compute_metrics(w, X, t)
        pbar.set_description_str(
            f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return all_ws, all_deltas, all_gradients


def is_neg_def(x):
    return np.all(np.linalg.eigvals(x) < 0)


def first_derivation(w, X, t, sigma_sq):
    # print(np.sum(X * (t - sigmoid(X @ w)), axis=0))
    result = (-1 / sigma_sq) * w + np.sum(X * (t - sigmoid(X @ w)),
                                          axis=0)[:, None]
    return result


w_start = np.random.normal(5, 5, size=(train_X.shape[1], 1))
assumed_sigma_sq = 1 / 1
learning_rate = 0.001

all_train_losses = []
all_val_losses = []
all_train_accs = []
all_val_accs = []
iterations = 100
all_ws_hats, all_deltas, all_gradients = stochastic_gradient_descent(
    train_X,
    train_y,
    w_start,
    assumed_sigma_sq,
    learning_rate,
    iterations,
    first_derivation,
)

## %%
for i in tqdm(range(len(all_ws_hats))):
    w_iter = all_ws_hats[i]
    m_train_acc, m_train_loss = compute_metrics(w_iter, train_X, train_y)
    m_val_acc, m_val_loss = compute_metrics(w_iter, val_X, val_y)
    all_train_losses.append(m_train_loss)
    all_val_losses.append(m_val_loss)
    all_train_accs.append(m_train_acc)
    all_val_accs.append(m_val_acc)

print(f"Final weights: ", all_ws_hats[-1].T)

## %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
burn_in_period = 0
plot_w_path_from_burnin(all_ws_hats[:15],
                        ax,
                        w_cov,
                        w_mu,
                        burn_in_period,
                        title="Diagonal",
                        precision=2)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))


plot_train_val_curve(smooth, all_train_losses, all_val_losses, ax, "Losses")
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ws = np.hstack(all_ws_hats).T
for i, wx in enumerate(ws.T[:, ::smooth]):
    ax.plot(wx, label=f"w{i}")
ax.set_xlabel("Iteration")
ax.set_ylabel("w")
ax.set_title("Weight Heights")
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
deltas = np.hstack(all_deltas).T / len(train_X)
for i, wx in enumerate(deltas.T[:, ::smooth]):
    ax.plot(wx, label=f"grad{i}")
ax.set_xlabel("Iteration")
ax.set_ylabel("grad")
ax.set_title("Weight Changes")
ax.legend()
plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(np.linalg.norm(deltas, axis=1)[::smooth], label=f"grad_len")
ax.set_xlabel("Iteration")
ax.set_ylabel("grad")
ax.legend()
plt.show()
# %%
