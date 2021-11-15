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
from tqdm import tqdm, tqdm_notebook

from logistic_regression_newton_rhapson import first_derivation, newton_method, second_derivation

# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True

# %%

# def true_function_sigmoid(x,y,w1,w2):
#     return 1 / (1 + np.exp(w1*x+w2*y+0.25))

# def observed_data_binary(d: int = 10, w1=2, w2=2, std=3):
#     data = np.random.uniform(-std, std, size=(d, 2))
#     # print(data)
#     x, y = data[:, 0], data[:, 1]
#     z = true_function_sigmoid(x, y, w1, w2)
#     # z_is_1 = true_function_sigmoid(x, y, w1, w2)
#     # z_is_0 = 1 - true_function_sigmoid(x, y, w1, w2)
#     # variance = np.random.uniform(shape=(n, 1))
#     # err = np.random.randn(d) * 0.25
#     # err = np.random.uniform(-1, 1) * 0.25
#     # z = z + err
#     # print(z.shape)
#     # print(err.shape)
#     return x, y, z < 0.5

n = 1000
w1, w2 = 0.1, -0.7
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
#       - log g(w; X; t; σ²) ≈ log g(w_hat; X; t; σ²) − v/2 (w-w_hat)²
#       - v is the negative of the second detivative of log g(w; X, t, σ²)


def laplace_approximation(
    w,
    X,
    t,
    sigma_sq,
    first_derivation,
    second_derivation,
    optimizer,
):

    for i in tqdm_notebook(range(iterations)):
        w, _, _, hessian = optimizer(
            w,
            X,
            t,
            sigma_sq,
            first_derivation,
            second_derivation,
        )
    hessian = second_derivation(w, X, t, sigma_sq)
    covariance = np.linalg.inv(-hessian)
    posterior = stats.multivariate_normal(w.flat, covariance)

    return posterior, w, covariance


def is_neg_def(x):
    return np.all(np.linalg.eigvals(x) < 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


w_hat = np.random.normal(1, 0.5, size=(train_X.shape[1], 1)) * 2
# w = np.zeros(shape=(train_X.shape[1], 1))
# w = np.array([[1], [-3], [3]])
# w = np.random.multivariate_normal([w1,w2], np.eye(train_X.shape[1]))[:, None]
# w = np.random.multivariate_normal([1, w1, w2], np.eye(train_X.shape[1]))[:, None]

all_ws = [w_hat]
all_deltas = []
all_train_accs = []
all_val_accs = []
assumed_sigma_sq = 1 / (n * 100)

# print("====================")
posterior, w_hat, w_cov_hat = laplace_approximation(
    w_hat,
    train_X,
    train_y,
    assumed_sigma_sq,
    first_derivation,
    second_derivation,
    newton_method,
)


def predict(posterior, val_X, num_samples=1000):
    ws = np.array([posterior.rvs()[:, None] for i in range(num_samples)]).reshape(num_samples, -2)
    logits = val_X @ ws.T
    probabilities = sigmoid(logits).mean(axis=1)
    return probabilities[:, None]


for i in range(10):
    w_sample = posterior.rvs()[:, None]
    # all_ws.append(w_sample)
    train_preds = predict(posterior, train_X, num_samples=1000)
    val_preds = predict(posterior, val_X, num_samples=1000)
    # train_losses = train_y - train_preds
    # val_losses = val_y - val_preds
    m_train_acc = np.mean(train_y == (train_preds > 0.5) * 1.0)
    m_val_acc = np.mean(val_y == ((val_preds > 0.5) * 1.0))
    # m_train_acc = np.mean(train_y == (train_preds>0.5)*1.0)
    # m_val_acc = np.mean(val_y == ((val_preds>0.5)*1.0))
    all_train_accs.append(m_train_acc)
    all_val_accs.append(m_val_acc)
    print(f"Accuracy Train: {m_train_acc:.4f} | Val: {m_train_acc:.4f}")

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.plot(all_train_losses[::smooth], label=f"train-loss")
# ax.plot(all_val_losses[::smooth], label=f"val-loss")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Loss")
# ax.set_title("Losses per iteration")
# ax.legend()
# plt.show()

## %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 10))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 10), sharex=True, sharey=True)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 10), sharex=True, sharey=True, subplot_kw={"projection": "3d"})
w_cov = np.array([[1, 0], [0, 1]])
w_mu = np.array([w1, w2])

X = np.linspace(w_mu[0] - w_cov[0, 0], w_mu[0] + w_cov[0, 0], 100)
Y = np.linspace(w_mu[1] - w_cov[1, 1], w_mu[1] + w_cov[1, 1], 100)
X, Y = np.meshgrid(X, Y)
distribution = stats.multivariate_normal(w_mu, w_cov)
Z_True = distribution.pdf(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

CS = ax1.contour(X, Y, Z_True)
# ax1.clabel(CS, inline=True, fontsize=10)
ax1.set_xlabel("w1")
ax1.set_ylabel("w2")
ax1.legend()
ax1.set_title("True Distribution")

X = np.linspace(w_hat[-2] - 2 * np.sqrt(w_cov_hat)[-2, -2], w_hat[-2] + 2 * np.sqrt(w_cov_hat)[-2, -2], 100)
# X = np.linspace(w_hat - w_hat*0.2, w_hat + w_hat*0.2, 100)
Y = np.linspace(w_hat[-1] - 2 * np.sqrt(w_cov_hat)[-1, -1], w_hat[-1] + 2 * np.sqrt(w_cov_hat)[-1, -1], 100)
# Y = np.linspace(w_hat - w_hat*0.2, w_hat + w_hat*0.2, 100)
X, Y = np.meshgrid(X, Y)
pred_distribution = stats.multivariate_normal(w_hat.flatten(), w_cov_hat)
# pred_distribution = stats.multivariate_normal(w_hat.flatten(), w_cov_hat)

Z_Pred = pred_distribution.pdf(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
w_sampled = pred_distribution.rvs(5)

CS = ax2.contour(X, Y, Z_Pred)
CS = ax1.contour(X, Y, Z_Pred)
ax1.scatter(w_sampled[:, -2], w_sampled[:, -1], s=100)
# ax1.scatter(Z_Pred[:, 0], Z_Pred[:, 1])
# ax2.clabel(CS, inline=True, fontsize=10)
ax2.set_xlabel("w1")
ax2.set_ylabel("w2")
# ax2.legend()
ax2.set_title("Approximation close-up")
# ax2.set_zlim3d(ax1.get_xlim3d())
# ax2.set_zlim3d(ax1.get_ylim3d())
# ax2.set_zlim3d(ax1.get_zlim3d())
# fig.tight_layout()
plt.show()
