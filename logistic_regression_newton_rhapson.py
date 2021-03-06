# %%
# https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/solving-logreg-newtons-method
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
#
# Newston Method
#   - Stopping Criterion # http://www-solar.mcs.st-and.ac.uk/~alan/MT2003/Numerical/node7.html
#   - Using any iterative scheme on a computer to estimate roots of an equation requires some condition to be satisfied so that the algorithm `knows'when to stop. There are various possibilities and these are listed below.
#   - $\vert x_{n+1} - x_{n}\vert$ sufficiently small. If the absolute value of two succesive rounded iterates agree to the same number of decimal places then $x_{n+1}$, the last estimate, is correct to that number of decimal places. Always take the last estimate as it is almost always more accurate. This raises an important point. Always keep two more decimal places in your calculations than the final answer needs. Thus, if the final result must be correct to 4 decimal places, then you should keep 6 decimal places in your workings.
#   - $\vert F(x_{n})\vert$ sufficiently small in some sense for some $x_{n}$. We are looking for the value of $x$ that makes $F(x) = 0$ so when $F(x)$ is sufficiently small we must be close to the root.
#   - A certain number of iterations have been performed. Stopping after, say 10 iterations, prevents the situation where the method has failed to converge and is aimlessly looping. This is not a problem when you use the method by hand as you will soon see if something has gone wrong but can cause problems when the method is implemented on a computer.
#   - Stop if $F^{\prime}(x_{n}) = 0$. This is extremely unlikely but if it does happen then computers do not like dividing by zero. It means the method has located a turning point of the function.


def newton_method(X, t, w_init, sigma_sq, num_iter, first_derivation,
                  second_derivation):

    num_features = X.shape[1]
    num_true_run_length = num_iter + 1
    all_ws = np.zeros((num_true_run_length, num_features, 1))
    all_deltas = np.zeros((num_true_run_length, num_features, 1))
    all_gradients = np.zeros((num_true_run_length, num_features, 1))
    all_hessians = np.zeros((num_true_run_length, num_features, num_features))
    w = w_init
    all_ws[0] = w_init
    all_hessians[0] = -np.eye(len(w)) * sigma_sq

    pbar = tqdm(range(1, num_true_run_length))
    for i in pbar:
        gradient = first_derivation(w, X, t, sigma_sq)
        hessian = second_derivation(w, X, t, sigma_sq)
        inv_hessian = np.linalg.inv(hessian)
        weight_change = (inv_hessian @ gradient)
        w = w - weight_change
        all_ws[i] = w
        all_deltas[i] = weight_change
        all_gradients[i] = gradient
        all_hessians[i] = hessian
        m_train_loss, m_train_acc = compute_metrics(w, X, t)
        pbar.set_description_str(
            f"Loss: {m_train_loss:.2f} | Acc: {m_train_acc:.2f}")

    return all_ws, all_deltas, all_gradients, all_hessians


def is_neg_def(x):
    return np.all(np.linalg.eigvals(x) < 0)


def first_derivation(w, X, t, sigma_sq):
    # print(np.sum(X * (t - sigmoid(X @ w)), axis=0))
    result = (-1 / sigma_sq) * w + np.sum(X * (t - sigmoid(X @ w)),
                                          axis=0)[:, None]
    return result


def second_derivation(w, X, t, sigma_sq):
    # block1 = (-1 / sigma_sq) * np.eye(len(w))
    # block2 = np.sum(X * X, axis=1)
    block2 = np.matmul(X[:, :, None], X[:, None, :])
    block3 = (sigmoid(X @ w) * (1 - sigmoid(X @ w)))[:, :, None]
    # return block1 - np.sum(block2 * block3, axis=0)
    # Andrew Ng Logistic Regression with Newton-Rhapson https://www.youtube.com/watch?v=fF-6QnVB-7E
    all_covs = block2 * block3
    all_regs = np.repeat([(-1 / sigma_sq) * np.eye(len(w))], len(X), axis=0)
    H = np.sum(all_regs - all_covs, axis=0)
    return H


def second_derivation_slow(w, X, t, sigma_sq):
    # According to Book by Rogers and Girolami
    num_features = len(w)
    H = np.zeros((num_features, num_features))
    for i in range(len(X)):
        x = X[i][None, :]
        xT = x.T
        block1 = (-1 / sigma_sq) * np.eye(num_features)
        block2 = xT @ xT.T
        P_n = sigmoid(x @ w)
        block3 = (P_n * (1 - P_n))
        H += block1 - block2 * block3

    return H


# def compute_metrics(w, X, y):
#     y_hat = sigmoid(X @ w)
#     losses = y - y_hat
#     m_loss = np.mean(np.abs(losses))
#     m_acc = np.mean(y == ((y_hat >= 0.5) * 1.0))
#     return m_loss, m_acc

w_start = np.random.normal(2, 2, size=(train_X.shape[1], 1))
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
    iterations,
    first_derivation,
    second_derivation,
)

# %%
for i in tqdm(range(len(all_ws_hats))):
    w_iter = all_ws_hats[i]
    m_train_acc, m_train_loss = compute_metrics(w_iter, train_X, train_y)
    m_val_acc, m_val_loss = compute_metrics(w_iter, val_X, val_y)
    all_train_losses.append(m_train_loss)
    all_val_losses.append(m_val_loss)
    all_train_accs.append(m_train_acc)
    all_val_accs.append(m_val_acc)

print(f"Final weights: ", all_ws_hats[-1].T)

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
burn_in_period = 0
plot_w_path_from_burnin(all_ws_hats,
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
