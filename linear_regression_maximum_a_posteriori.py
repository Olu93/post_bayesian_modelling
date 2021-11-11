# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import meshgrid
from numpy.random import multivariate_normal
import pandas as pd
from IPython.display import display
from scipy import stats
from matplotlib import cm
import random as r
from data import observed_data, observed_data_linear, true_function_polynomial
from helper import add_bias_vector, create_polinomial_bases

np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
# %%
n = 50
exp_1, exp_2 = 2, 3
val_n = 50
p_ord = 2
data = np.vstack(np.array(observed_data(n, exp_1, exp_2)).T)
val_data = np.vstack(np.array(observed_data(val_n, exp_1, exp_2)).T)
display(data.max(axis=0))
display(data.min(axis=0))
display(data.mean(axis=0))

# %%
# %%
train_X = add_bias_vector(create_polinomial_bases(data[:, :-1], order=p_ord))
train_y = data[:, -1][:, None]
val_X = add_bias_vector(create_polinomial_bases(val_data[:, :-1], order=p_ord))
val_y = val_data[:, -1][:, None]
num_features = train_X.shape[1]

train_X.shape
# %%
# # Bayesian Treatment: Requires selecting Likelihood and Prior and estimating predictive distribution
# Likelihood:               p(t|X,w,σ²) = N(Xw; σ²I) = t_n = Xw + ε
# - Random noise ε ~ N(0,1)
# Prior:                    p(w|∆) = p(w|µ0, Σ0) = N(µ0; Σ0)
# - ∆ is a set of params
# - Needs to be Gaussian for exact posterior estimation
# Posterior:                p(w|t; X; σ2; µ0; Σ0) = N(µw; Σw)
# - We want to estimate the new µw; Σw
# Predictive distribution:  p(t_new|x_new,X,t,σ2,∆) = ∫ p(t_new|x_new,w,σ2)p(w|t,X,σ2,∆) dw
# - New t not dependent on old X
# - New t not dependent on ∆ - Only used for generating w
# - w not dependent on new x
# - Has w integrated out -  For this we can use expectation of w with respect to posterior
# -- p(t_new|x_new; w; σ²) = N(x_new.T @ w; σ²)
# -- With w integrated out N(x_new.T @ w; σ²) = N (x_new.T @ µw; σ2 + x_new.T @ Σw @ x_new)
# Marginal Likelihood:     p(t|X; µ0; Σ0) = ∫ p(t|X,w,σ²) * p(w|∆) dw  = N(X @ µ0; σ²*I + X @ Σ0 @ X.T)


def compute_posterior_covariance(X, sigma_sq, init_w_cov):
    # covariance is the covariance of the normal distribution we assume for the posterior
    posterior_covariance = np.linalg.inv((1 / sigma_sq) * X.T @ X + np.linalg.inv(init_w_cov))
    return posterior_covariance


def compute_posterior_mean(X, y, sigma_sq, posterior_cov, init_w_means, init_w_cov):
    # mean is the mean of the normal distribution we assume for the posterior
    posterior_means = posterior_cov @ ((1 / sigma_sq) * X.T @ y + np.linalg.inv(init_w_cov) @ init_w_means)
    return posterior_means


def compute_sigma_squared(X, y, w):
    # Same as (sum((y_n - x_n*w)²))/N = ((y-Xw).T @ (y-Xw))/N
    total_sigma = y.T @ y - y.T @ X @ w
    expected_sigma = total_sigma / len(y)
    return expected_sigma


def compute_marginal_likelihood(X, y, sigma_sq, init_w_means, init_w_cov):
    # covariance is the covariance of the normal distribution we assume for the posterior

    new_mu = X @ init_w_means
    new_cov = X @ init_w_cov @ X.T + sigma_sq * np.eye(X.shape[0])
    likelihoods = stats.norm(new_mu, new_cov).pdf(y)  # Training y
    log_likelihoods = np.log(likelihoods)
    sum_log_likelihoods = log_likelihoods.sum()
    return sum_log_likelihoods


# Next predictions if you have an estimate for w and sigma_squared
def next_predictions(x_new, X, w_pred, sigma_squared_pred):
    t_new = x_new @ w_pred
    cov_w_pred = sigma_squared_pred * np.linalg.inv(X.T @ X)
    sigma_squared_new = ((x_new @ cov_w_pred) * x_new).sum(axis=1)

    return t_new, sigma_squared_new[:, None]


assumed_sigma_sq = 10
init_w_cov = np.eye(num_features)
init_w_means = np.zeros((num_features, 1))
posterior_w_covariance = compute_posterior_covariance(train_X, assumed_sigma_sq, init_w_cov)
posterior_w_means = compute_posterior_mean(train_X, train_y, assumed_sigma_sq, posterior_w_covariance, init_w_means,
                                           init_w_cov)

posterior_w_covariance, posterior_w_means
# %%
w_sampled = np.random.multivariate_normal(posterior_w_means.squeeze(), posterior_w_covariance)
sigma_squared_pred = compute_sigma_squared(train_X, train_y, w_sampled)
w_sampled, np.sqrt(sigma_squared_pred)

# %%


# Shows the function of the predicted weights and their
def plot_weights(p_ord, selected_features, train_X, plot_X, w_sampled, sigma_squared_pred, ax):
    u = plot_X[selected_features[0]]
    v = plot_X[selected_features[1]]
    u, v = np.meshgrid(u, v)

    plot_X_flat = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
    plot_X_polym = add_bias_vector(create_polinomial_bases(plot_X_flat, order=p_ord))
    y_new, sigma_squared_new = next_predictions(plot_X_polym, train_X, w_sampled, sigma_squared_pred)

    # print(sigma_squared_new.T)
    sigma = np.nan_to_num(np.sqrt(sigma_squared_new))
    plot = ax.scatter(plot_X_polym[:, 1], plot_X_polym[:, 2], y_new.squeeze(), c=sigma.squeeze())
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    return plot


selected_features = ["feat_1", "feat_2"]
grid_X = np.random.uniform(-10, 10, (100, num_features))
plot_X = pd.DataFrame(grid_X, columns=[f"feat_{ft}" for ft in range(num_features)])
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
plot = plot_weights(p_ord, selected_features, train_X, plot_X, w_sampled, sigma_squared_pred, ax=ax)
fig.colorbar(plot, shrink=0.5, aspect=5, label="Standard Deviation")
# %%
fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={"projection": "3d"})
init_w_cov = np.eye(num_features)
init_w_means = np.zeros((num_features, 1))
batch_size = len(train_X) // 20
extend_y = (train_y.max() - train_y.min())
all_sampled_weights = []
assumed_sigma_sq = 5
for i, ax in zip(range(1, 11), axes.flatten()):
    train_X_subset = train_X[:i * batch_size]
    train_y_subset = train_y[:i * batch_size]
    posterior_w_covariance = compute_posterior_covariance(train_X_subset, assumed_sigma_sq, init_w_cov)
    posterior_w_means = compute_posterior_mean(train_X_subset, train_y_subset, assumed_sigma_sq, posterior_w_covariance,
                                               init_w_means, init_w_cov)
    # w_sampled = np.random.multivariate_normal(posterior_w_means.squeeze(), posterior_w_covariance)
    all_sampled_weights.append(w_sampled)
    plot = plot_weights(p_ord, selected_features, train_X_subset, plot_X, posterior_w_means, assumed_sigma_sq, ax=ax)
    init_w_cov = posterior_w_covariance
    init_w_means = posterior_w_means

plt.show()


# %%
# Next mus and sigmas for new x. Each x is a probability distribution
def posterior_predictive_ditribution(X_new, posterior_means, posterior_cov, sigma_squared):
    X_new = X_new.values
    new_mus = X_new @ posterior_means
    new_covs = sigma_squared + ((X_new @ posterior_cov) * X_new).sum(axis=1)
    return new_mus, new_covs[:, None]


def plot_posterior_predictive(ys, mu, sigma_suqared, ax):
    normal_dist = stats.norm(mu, np.sqrt(sigma_suqared))
    values = normal_dist.pdf(ys)
    ax.plot(ys, values)
    ax.set_xlabel('y_new')
    ax.set_ylabel('p(y_new|x_new, X, y, σ²)')
    return ax


selected = 1
assumed_sigma_sq = 10
plot_X = pd.DataFrame(val_X, columns=[f"feat_{ft}" for ft in range(num_features)])
init_w_cov = np.eye(num_features)
init_w_means = np.zeros((num_features, 1))
posterior_w_covariance = compute_posterior_covariance(train_X, assumed_sigma_sq, init_w_cov)
posterior_w_means = compute_posterior_mean(train_X, train_y, assumed_sigma_sq, posterior_w_covariance, init_w_means,
                                           init_w_cov)
mus, sigma_squares = posterior_predictive_ditribution(plot_X, posterior_w_means, posterior_w_covariance,
                                                      assumed_sigma_sq)
selected_mu, selected_sigma_square = mus[selected], sigma_squares[selected]
spread = 2 * selected_mu
lower_lim = selected_mu - spread
upper_lim = selected_mu + spread
ys_to_plot = np.linspace(lower_lim, upper_lim, 100)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()
plot_posterior_predictive(ys_to_plot, selected_mu, selected_sigma_square, ax=ax)
plt.show()


# %%
def plot_posterior_predictive_change(num_data, selected, exp_y_mean, exp_y_var, ax, precision=100):
    sigma = np.sqrt(exp_y_var)[selected]
    mu = exp_y_mean[selected]
    # spread = 3 * sigmas
    lower_lims = mu - spread
    upper_lims = mu + spread
    xs = np.linspace(lower_lims.squeeze(), upper_lims.squeeze(), precision)
    ys = stats.norm(mu, sigma).pdf(xs)

    plot = ax.scatter(num_data, xs, ys, antialiased=True, alpha=1)
    ax.set_xlabel('mus')
    ax.set_ylabel('y_new')
    ax.set_zlabel('p(y_new|x_new, X, y, σ²)')
    # ax.view_init(45, 30)
    return plot


assumed_sigma_sq = 10
init_w_cov = np.eye(num_features)
init_w_means = np.zeros((num_features, 1))
num_steps = 10
batch_size = len(train_X) // (3 * num_steps)
plot_X = pd.DataFrame(val_X, columns=[f"feat_{ft}" for ft in range(num_features)])
plot_X_sorted = plot_X.sort_values(plot_X.columns[selected])
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
for i in range(1, num_steps + 1):
    train_X_subset = train_X[:i * batch_size]
    train_y_subset = train_y[:i * batch_size]
    posterior_w_covariance = compute_posterior_covariance(train_X_subset, assumed_sigma_sq, init_w_cov)
    posterior_w_means = compute_posterior_mean(train_X_subset, train_y_subset, assumed_sigma_sq, posterior_w_covariance,
                                               init_w_means, init_w_cov)
    y_new_mus, y_new_sigma_squares = posterior_predictive_ditribution(plot_X, posterior_w_means, posterior_w_covariance,
                                                                      assumed_sigma_sq)
    plot_posterior_predictive_change(len(train_y_subset), 1, y_new_mus, y_new_sigma_squares, ax)
plt.show()

# %%


def plot_posterior_predictive_multivariate(mus, sigmas_squared, ax, num_elems=10, with_mid_line=False):
    sigmas = np.sqrt(sigmas_squared)
    spread = 2 * sigmas
    lower_lims = mus - spread
    upper_lims = mus + spread
    data = pd.DataFrame(np.hstack([mus, sigmas, lower_lims, upper_lims]))
    # data = data.sort_values(0).reset_index().drop('index', axis=1)

    display(data)
    # x = range(len(mus))
    # ax.scatter(xs, z)
    # ax.scatter(xs, ys)
    ax.set_xlabel('data_point')
    ax.set_ylabel('mus')
    if with_mid_line:
        ax.plot(data.index, data[0])
    ax.fill_between(data.index, data[2], data[3], color='b', alpha=.1)


assumed_sigma_sq = 10
init_w_cov = np.eye(num_features)
init_w_means = np.zeros((num_features, 1))
fig = plt.figure(figsize=(15, 5))
selected = 1
# grid_X = np.repeat(np.linspace(-5, 5, 10)[:, None], num_features, axis=1)
# grid_X = np.linspace(-5, 5, 10)[:, None]
grid_X = np.random.uniform(-5, 5, 10)
grid_X = grid_X[grid_X.argsort()][:, None]
grid_X = np.array(np.meshgrid(grid_X, grid_X)).reshape(2,-1).T
# grid_X[:, [3]] = 0
# plot_X_polym = add_bias_vector(create_polinomial_bases(np.hstack(([grid_X, grid_X])), order=p_ord))
plot_X_polym = add_bias_vector(create_polinomial_bases(grid_X, order=p_ord))
plot_X = pd.DataFrame(plot_X_polym, columns=[f"feat_{ft}" for ft in range(num_features)])
# plot_data = pd.DataFrame(np.hstack([val_X, val_y]), columns=[f"feat_{ft}" for ft in range(num_features)] + ["target"]).iloc[:10]
# plot_data_sorted = plot_data.sort_values(plot_data.columns[selected])
# plot_X = plot_data_sorted.iloc[:, :-1]
ax = fig.add_subplot()
posterior_w_covariance = compute_posterior_covariance(train_X, assumed_sigma_sq, init_w_cov)
posterior_w_means = compute_posterior_mean(train_X, train_y, assumed_sigma_sq, posterior_w_covariance, init_w_means,
                                           init_w_cov)

mus, sigma_squares = posterior_predictive_ditribution(plot_X, posterior_w_means, posterior_w_covariance,
                                                      assumed_sigma_sq)
# xs, ys = plot_data_sorted.iloc[:, selected], plot_data_sorted.iloc[:, -1]
xs = range(len(plot_X))
# xs.iloc[:, [2]]
plot_posterior_predictive_multivariate(mus, sigma_squares, ax, with_mid_line=True)

plt.show()
# %%
