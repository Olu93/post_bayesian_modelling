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
from helper import add_bias_vector, compute_metrics, create_polinomial_bases, predict, sigmoid
from tqdm.notebook import tqdm
from logistic_regression_newton_rhapson import first_derivation, newton_method, second_derivation
from viz import plot_train_val_curve, plot_w_samples

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
noise = 0.1
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
#       - log g(w; X; t; σ²) ≈ log g(w_hat; X; t; σ²) − v/2 (w-w_hat)²
#       - v is the negative of the second detivative of log g(w; X, t, σ²)


def laplace_approximation(
    w_init,
    X,
    t,
    sigma_sq,
    iterations,
    first_derivation,
    second_derivation,
    optimizer,
):
    w = w_init
    all_ws_hats, all_deltas, all_gradients, all_hessians = optimizer(
        X,
        t,
        w,
        sigma_sq,
        iterations,
        first_derivation,
        second_derivation,
    )
    optimization_summaries = (all_ws_hats, all_deltas, all_gradients,
                              all_hessians)
    w = all_ws_hats[-1]
    hessian = all_hessians[-1]
    covariance = np.linalg.inv(-hessian)
    posterior = stats.multivariate_normal(w.flat, covariance)

    return posterior, w, covariance, optimization_summaries


w_start = np.random.normal(2, 2, size=(train_X.shape[1], 1))
assumed_sigma_sq = 1 / 100

all_train_losses = []
all_val_losses = []
all_train_accs = []
all_val_accs = []

iterations = 20
posterior, w_hat, w_cov_hat, summaries = laplace_approximation(
    w_start,
    train_X,
    train_y,
    assumed_sigma_sq,
    iterations,
    first_derivation,
    second_derivation,
    newton_method,
)

num_samples = 100

all_posteriors_through_optimization = [
    stats.multivariate_normal(summaries[0][i].flat,
                              np.linalg.inv(-summaries[-1][i]))
    for i in range(len(summaries[0]))
]
all_ws_hats = [
    historical_posterior.rvs(num_samples).mean(axis=0)
    for historical_posterior in all_posteriors_through_optimization
]
all_ws_hats = np.array(all_ws_hats)
print(f"Final weights: {all_ws_hats[-1].T}")
# %%
for i in tqdm(range(len(all_ws_hats))):
    w_iter = all_ws_hats[i][:, None]
    m_train_acc, m_train_loss = compute_metrics(w_iter, train_X, train_y)
    m_val_acc, m_val_loss = compute_metrics(w_iter, val_X, val_y)
    all_train_losses.append(m_train_loss)
    all_val_losses.append(m_val_loss)
    all_train_accs.append(m_train_acc)
    all_val_accs.append(m_val_acc)

fig, (ax1) = plt.subplots(1, 1, figsize=(7, 10))
plot_train_val_curve(1, all_train_losses, all_val_losses, ax1, "Losses")
# %%
fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))
burn_in_period = 0
plot_w_samples(all_posteriors_through_optimization[-1].rvs(1000),
               ax1,
               w_cov,
               w_mu,
               burn_in_period,
               title="Over the history of optimization iterations",
               precision=2)
fig.tight_layout()
plt.show()
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 14))


def plot_w_contour(w_mu, w_cov, title, ax, point=None):
    X_true = np.linspace(w_mu[0] - w_cov[0, 0], w_mu[0] + w_cov[0, 0], 100)
    Y_true = np.linspace(w_mu[1] - w_cov[1, 1], w_mu[1] + w_cov[1, 1], 100)
    X, Y = np.meshgrid(X_true, Y_true)
    distribution = stats.multivariate_normal(w_mu, w_cov)
    Z_True = distribution.pdf(np.array([X.flatten(),
                                        Y.flatten()]).T).reshape(X.shape)
    CS = ax.contour(X, Y, Z_True)
    if point is not None:
        ax.scatter(point[0], point[1])
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.legend()
    ax.set_title(title)


plot_w_contour(w_mu, w_cov, "True Distribution", ax1, w_hat)
plot_w_contour(w_hat.flat, w_cov_hat, "Approximation close-up", ax2)

fig.tight_layout()
plt.show()

# %%
