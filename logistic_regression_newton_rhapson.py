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
from helper import add_bias_vector, create_polinomial_bases, sigmoid
from tqdm.notebook import tqdm
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
iterations = 1000
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
#
# Newston Method
#   - Stopping Criterion # http://www-solar.mcs.st-and.ac.uk/~alan/MT2003/Numerical/node7.html
#   - Using any iterative scheme on a computer to estimate roots of an equation requires some condition to be satisfied so that the algorithm `knows'when to stop. There are various possibilities and these are listed below.
#   - $\vert x_{n+1} - x_{n}\vert$ sufficiently small. If the absolute value of two succesive rounded iterates agree to the same number of decimal places then $x_{n+1}$, the last estimate, is correct to that number of decimal places. Always take the last estimate as it is almost always more accurate. This raises an important point. Always keep two more decimal places in your calculations than the final answer needs. Thus, if the final result must be correct to 4 decimal places, then you should keep 6 decimal places in your workings.
#   - $\vert F(x_{n})\vert$ sufficiently small in some sense for some $x_{n}$. We are looking for the value of $x$ that makes $F(x) = 0$ so when $F(x)$ is sufficiently small we must be close to the root.
#   - A certain number of iterations have been performed. Stopping after, say 10 iterations, prevents the situation where the method has failed to converge and is aimlessly looping. This is not a problem when you use the method by hand as you will soon see if something has gone wrong but can cause problems when the method is implemented on a computer.
#   - Stop if $F^{\prime}(x_{n}) = 0$. This is extremely unlikely but if it does happen then computers do not like dividing by zero. It means the method has located a turning point of the function.


def newton_method(X, t, w_init, sigma_sq, num_iter, first_derivation, second_derivation):

    num_features = X.shape[1]
    indices = np.random.randint(0, len(X), size=num_iter)
    all_ws = np.zeros((num_iter + 1, num_features, 1))
    all_deltas = np.zeros((num_iter + 1, num_features, 1))
    all_gradients = np.zeros((num_iter + 1, num_features, 1))
    all_hessians = np.zeros((num_iter + 1, num_features, num_features))
    w = w_init
    all_ws[0] = w_init

    pbar = tqdm(range(1, num_iter))
    for i in pbar:
        selected_datapoint = indices[i - 1]
        x_n = X[selected_datapoint, :][None, :]
        t_n = t[selected_datapoint, :]
        gradient = first_derivation(w, x_n, t_n, sigma_sq)
        hessian = second_derivation(w, x_n, t_n, sigma_sq)
        weight_change = (np.linalg.inv(hessian) @ gradient)  # / (len(X) if use_mean else 1)
        w = w - weight_change
        all_ws[i] = w
        all_deltas[i] = weight_change
        all_gradients[i] = gradient
        all_hessians[i] = hessian
        m_train_acc, m_train_loss = compute_metrics(w, X, t)
        pbar.set_description_str(f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return all_ws, all_deltas, all_gradients, all_hessians


def is_neg_def(x):
    return np.all(np.linalg.eigvals(x) < 0)


def first_derivation(w, X, t, sigma_sq):
    # print(np.sum(X * (t - sigmoid(X @ w)), axis=0))
    result = (-1 / sigma_sq) * w + np.sum(X * (t - sigmoid(X @ w)), axis=0)[:, None]
    return result


# def second_derivation(w, X, t, sigma_sq):
#     # block1 = (-1 / sigma_sq) * np.eye(len(w))
#     # block2 = np.sum(X * X, axis=1)
#     block2 = np.matmul(X[:, :, None], X[:, None, :])
#     block3 = (sigmoid(X @ w) * (1 - sigmoid(X @ w)))[:, :, None]
#     # return block1 - np.sum(block2 * block3, axis=0)
#     # Andrew Ng Logistic Regression with Newton-Rhapson https://www.youtube.com/watch?v=fF-6QnVB-7E
#     H = np.mean(block2 * block3, axis=0)
#     return (-1 / sigma_sq) * np.eye(len(w)) - H


def second_derivation(w, x, t, sigma_sq):
    # According to Book by Rogers and Girolami
    x = x.T
    block1 = (-1 / sigma_sq) * np.eye(len(w))
    block2 = x @ x.T
    P_n = sigmoid(x.T @ w)
    block3 = (P_n * (1 - P_n))
    return block1 - block2 * block3


def compute_metrics(w, X, y):
    y_hat = sigmoid(X @ w)
    losses = y - y_hat
    m_loss = np.mean(np.abs(losses))
    m_acc = np.mean(y == ((y_hat >= 0.5) * 1.0))
    return m_loss, m_acc


w_start = np.random.uniform(10, 11, size=(train_X.shape[1], 1))
assumed_sigma_sq = 1 / 100

all_train_losses = []
all_val_losses = []
all_train_accs = []
all_val_accs = []

all_ws_hats, all_deltas, all_gradients, all_hessians = newton_method(
    train_X,
    train_y,
    w_start,
    assumed_sigma_sq,
    100,
    first_derivation,
    second_derivation,
)

for i in range(len(all_ws_hats)):
    w_iter = all_ws_hats[i]
    m_train_acc, m_train_loss = compute_metrics(w_iter, train_X, train_y)
    m_val_acc, m_val_loss = compute_metrics(w_iter, val_X, val_y)
    all_train_losses.append(m_train_loss)
    all_val_losses.append(m_val_loss)
    all_train_accs.append(m_train_acc)
    all_val_accs.append(m_val_acc)

# with tqdm(range(iterations)) as pbar:
#     for i in pbar:
#         # print("====================")
#         w, w_delta, gradient, hessian = newton_method_vectorised(
#             w,
#             train_X,
#             train_y,
#             assumed_sigma_sq,
#             first_derivation,
#             second_derivation,
#         )
#         all_ws.append(w)
#         all_deltas.append(w_delta)
#         m_train_acc, m_val_acc, m_train_loss, m_val_loss = compute_metrics(train_X, train_y, val_X, val_y, w)
#         all_train_losses.append(m_train_acc)
#         all_val_losses.append(m_val_acc)
#         pbar.set_description_str(f"Train: {m_train_acc:.4f} | Val: {m_val_acc:.4f}")
print(f"Final weights: ", all_ws_hats[-1])

## %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
burn_in_period = 20
all_ws_hats = np.hstack(all_ws_hats).T[::1]


def plot_w_path(all_w_hats, ax, w_cov, w_mu, burn_in_period, title="", precision=2):
    all_w_hats_to_use = all_w_hats[burn_in_period:]
    x_min, y_min = all_w_hats_to_use.min(axis=0)
    x_max, y_max = all_w_hats_to_use.max(axis=0)
    x_cov = precision * np.sqrt(w_cov[0, 0])
    y_cov = precision * np.sqrt(w_cov[1, 1])
    x_mu = w_mu[0]
    y_mu = w_mu[1]
    x_lims = np.min([x_min, x_mu - x_cov]), np.max([x_max, x_mu + x_cov])
    y_lims = np.min([y_min, y_mu - y_cov]), np.max([y_max, y_mu + y_cov])
    X = np.linspace(x_lims[0], x_lims[1], 100)
    Y = np.linspace(y_lims[0], y_lims[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z_True = stats.multivariate_normal(w_mu, w_cov).pdf(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

    CS = ax.contour(X, Y, Z_True)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.plot(all_w_hats_to_use[:, -2], all_w_hats_to_use[:, -1])
    ax.scatter(all_w_hats_to_use[:, -2], all_w_hats_to_use[:, -1], s=10, c="blue", label="step")
    ax.scatter(all_w_hats_to_use[0][-2], all_w_hats_to_use[0][-1], s=100, c='green', label="start")
    ax.scatter(all_w_hats_to_use[-1][-2], all_w_hats_to_use[-1][-1], s=100, c='red', label="end")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_title(f"Weight Movement: {title}")
    ax.legend()


plot_w_path(all_ws_hats, ax, w_cov, w_mu, burn_in_period, title="Diagonal", precision=2)
where_it_is_to_much = np.where(np.abs(all_ws_hats).sum(axis=1) > 100)
print(f"Where the vals are too high: {where_it_is_to_much}")
# X = np.linspace(w1 - contour_width, w1 + contour_width, 100)
# Y = np.linspace(w2 - contour_width, w2 + contour_width, 100)
# X, Y = np.meshgrid(X, Y)
# distribution = stats.multivariate_normal(w_mu, w_cov)
# Z = distribution.pdf(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

# CS = ax.contour(X, Y, Z)
# ax.clabel(CS, inline=True, fontsize=10)

# ws = np.hstack(all_ws).T[::1]
# w1s, w2s = ws[:, -2], ws[:, -1]
# zws = distribution.pdf(ws[:, [-2, -1]])
# ax.plot(w1s, w2s)
# ax.scatter(w1s[0], w2s[0], s=100, c='green', label="start")
# ax.scatter(w1s[-1], w2s[-1], s=100, c='red', label="end")
# ax.set_xlabel("w1")
# ax.set_ylabel("w2")
# # ax.view_init(0, 15)
# ax.legend()
# ax.set_title("Weight movement")
# plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(all_train_losses[::smooth], label=f"train-loss")
ax.plot(all_val_losses[::smooth], label=f"val-loss")
ax.set_xlabel("Iteration")
ax.set_ylabel("Acc")
ax.set_title("Accuracies per iteration")
ax.legend()
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
