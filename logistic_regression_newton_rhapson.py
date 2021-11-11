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
from data import observed_data, observed_data_binary, observed_data_linear, true_function
from helper import add_bias_vector, create_polinomial_bases
from tqdm import tqdm
# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True
# %%
n = 500
w1, w2 = -2, -1
xstd = 10
val_n = 100
p_ord = 1
iterations = 300 if IS_EXACT_FORMULA else 1000
smooth = 1 if IS_EXACT_FORMULA else 1
data = np.vstack(np.array(observed_data_binary(n, w1, w2, xstd)).T)
val_data = np.vstack(np.array(observed_data_binary(val_n, w1, w2, xstd)).T)
display(data.max(axis=0))
display(data.min(axis=0))
display(data.mean(axis=0))

train_X = data[:, :-1]
train_X = add_bias_vector(create_polinomial_bases(data[:, :-1], p_ord))
train_y = data[:, -1][:, None]
val_X = val_data[:, :-1]
val_X = add_bias_vector(create_polinomial_bases(val_data[:, :-1], p_ord))
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
#       -


def newton_method(x, first_derivation, second_derivation):
    x_opt = x - (first_derivation(x) / second_derivation(x))
    return x_opt


def newton_method_vectorised(
    w,
    X,
    t,
    sigma_sq,
    first_derivation,
    second_derivation,
    use_mean=False,
):
    # print("init w")
    # print(w)
    gradient = first_derivation(w, X, t, sigma_sq)
    # print("opt")
    # print(opt)
    hessian = np.linalg.inv(second_derivation(gradient, X, t, sigma_sq))
    # print("hessian")
    # print(hessian)

    # print("update amount")
    # print(hessian @ opt)
    weight_change = (hessian @ gradient) / (len(X) if use_mean else 1)
    w_new = w - weight_change
    # w_new = w - (hessian @ gradient) #/ (len(X) if use_mean else 1)
    # print("w_opt")
    # print(w_opt)
    return w_new, weight_change, gradient, hessian


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def first_derivation(w, X, t, sigma_sq):
    # print(np.sum(X * (t - sigmoid(X @ w)), axis=0))
    result = (-1 / sigma_sq) * w + np.sum(X * (t - sigmoid(X @ w)), axis=0)[:, None]
    return result


def second_derivation(w, X, t, sigma_sq):
    block1 = (-1 / sigma_sq) * np.eye(len(w))
    # block2 = np.sum(X * X, axis=1)
    block2 = X * X
    block3 = sigmoid(X @ w) * (1 - sigmoid(X @ w))
    return block1 - np.sum(block2 * block3, axis=0)


# w = np.random.normal(1, 0.5, size=(train_X.shape[1], 1))
w = np.zeros(shape=(train_X.shape[1], 1))
# w = np.random.multivariate_normal([w1,w2], np.eye(train_X.shape[1]))[:, None]
# w = np.random.multivariate_normal([1, w1, w2], np.eye(train_X.shape[1]))[:, None]

all_ws = [w]
all_deltas = []
all_train_losses = []
all_val_losses = []
assumed_sigma_sq = 1/10
with tqdm(range(iterations)) as pbar:
    for i in pbar:
        # print("====================")
        w, w_delta, gradient, hessian = newton_method_vectorised(
            w,
            train_X,
            train_y,
            assumed_sigma_sq,
            first_derivation,
            second_derivation,
            ~IS_EXACT_FORMULA,
        )
        all_ws.append(w)
        all_deltas.append(w_delta)
        train_preds = train_X @ w
        val_preds = val_X @ w
        train_losses = train_y - train_preds
        val_losses = val_y - val_preds
        m_train_loss = np.mean(np.abs(train_losses))
        m_val_loss = np.mean(np.abs(val_losses))
        all_train_losses.append(m_train_loss)
        all_val_losses.append(m_val_loss)
        pbar.set_description_str(f"Train: {m_train_loss:.4f} | Val: {m_train_loss:.4f}")
print(f"Final weights: ", w.T)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(all_train_losses[::smooth], label=f"train-loss")
ax.plot(all_val_losses[::smooth], label=f"val-loss")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Losses per iteration")
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
w_cov = np.array([[1, 0], [0, 1]])
w_mu = np.array([w1, w2])
contour_width = 3

X = np.linspace(w1 - contour_width, w1 + contour_width, 100)
Y = np.linspace(w2 - contour_width, w2 + contour_width, 100)
X, Y = np.meshgrid(X, Y)
distribution = stats.multivariate_normal(w_mu, w_cov)
Z = distribution.pdf(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)

ws = np.hstack(all_ws).T[::smooth]
w1s, w2s = ws[:, -2], ws[:, -1]
ax.plot(w1s, w2s)
ax.scatter(w1s[0], w2s[0], s=100, c='green', label="start")
ax.scatter(w1s[-1], w2s[-1], s=100, c='red', label="end")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.legend()
ax.set_title("Weight movement")
plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ws = np.hstack(all_ws).T
for i, wx in enumerate(ws.T[:, ::smooth]):
    ax.plot(wx, label=f"w{i}")
ax.set_xlabel("Iteration")
ax.set_ylabel("w")
ax.set_title("Weight Heights")
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
deltas = np.hstack(all_deltas).T/len(train_X)
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
