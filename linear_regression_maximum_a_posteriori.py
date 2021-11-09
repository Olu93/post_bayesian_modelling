# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import meshgrid
import pandas as pd
from IPython.display import display
from scipy import stats
from matplotlib import cm
import random as r
from data import observed_data, observed_data_linear, true_function
from helper import add_bias_vector, create_polinomial_bases

np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
# %%
n = 50
val_n = 10
p_ord = 2
data = np.vstack(np.array(observed_data(n, 1, 2)).T)
val_data = np.vstack(np.array(observed_data(val_n, 1, 2)).T)
display(data.max(axis=0))
display(data.min(axis=0))
display(data.mean(axis=0))

# %%
# %%
X, y = data[:, :-1], data[:, -1].reshape(-1, 1)
train_X = add_bias_vector(create_polinomial_bases(X, order=p_ord))
val_X = add_bias_vector(create_polinomial_bases(data[:, :-1], order=p_ord))
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
    new_cov = X@init_w_cov@X.T + sigma_sq * np.eye(X.shape[0])
    likelihoods = stats.norm(new_mu, new_cov).pdf(y) # Training y
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
init_w_cov = np.eye(train_X.shape[1])
init_w_means = np.zeros((train_X.shape[1], 1))
posterior_covariance = compute_posterior_covariance(train_X, assumed_sigma_sq, init_w_cov)
posterior_means = compute_posterior_mean(train_X, y, assumed_sigma_sq, posterior_covariance, init_w_means, init_w_cov)

posterior_covariance, posterior_means
# %%
w_sampled = np.random.multivariate_normal(posterior_means.squeeze(), posterior_covariance)
sigma_squared_pred = compute_sigma_squared(train_X, y, w_sampled)
w_sampled, np.sqrt(sigma_squared_pred)

# %%


# Shows the function of the predicted weights and their
def plot_weights_given_new_data(p_ord, selected_features, train_X, plot_X, w_sampled, sigma_squared_pred, ax):
    u = plot_X[selected_features[0]]
    v = plot_X[selected_features[1]]
    u, v = np.meshgrid(u, v)

    plot_X_flat = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
    plot_X_polym = add_bias_vector(create_polinomial_bases(plot_X_flat, order=p_ord))
    y_new, sigma_squared_new = next_predictions(plot_X_polym, train_X, w_sampled, sigma_squared_pred)
    ax.scatter(plot_X_polym[:, 1], plot_X_polym[:, 2], y_new.squeeze(), c=np.sqrt(sigma_squared_new).squeeze())


selected_features = ["feat_1", "feat_2"]
grid_X = np.random.uniform(-3, 3, (100, X.shape[1]))
plot_X = pd.DataFrame(val_X, columns=[f"feat_{ft}" for ft in range(val_X.shape[1])])
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
plot_weights_given_new_data(p_ord, selected_features, train_X, plot_X, w_sampled, sigma_squared_pred, ax=ax)
# %%
fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={"projection": "3d"})
init_w_cov = np.eye(train_X.shape[1])
init_w_means = np.zeros((train_X.shape[1], 1))
batch_size = len(train_X) // 10
extend_y = (y.max() - y.min())
limit_padding = int(extend_y * 0.5)
min_z = y.min() - limit_padding
max_z = y.max() + limit_padding
all_sampled_weights = []
for i, ax in zip(range(1, 11), axes.flatten()):
    assumed_sigma_sq = 5
    train_X_subset = train_X[:i * 2]
    train_y_subset = y[:i * 2]
    posterior_covariance = compute_posterior_covariance(train_X_subset, assumed_sigma_sq, init_w_cov)
    posterior_means = compute_posterior_mean(train_X_subset, train_y_subset, assumed_sigma_sq, posterior_covariance,
                                             init_w_means, init_w_cov)
    w_sampled = np.random.multivariate_normal(posterior_means.squeeze(), posterior_covariance)
    all_sampled_weights.append(w_sampled)
    plot_weights_given_new_data(p_ord, selected_features, train_X_subset, plot_X, w_sampled, assumed_sigma_sq, ax=ax)
    ax.set_zlim3d(bottom=min_z, top=max_z)
    init_w_cov = posterior_covariance
    init_w_means = posterior_means
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
init_w_cov = np.eye(train_X.shape[1])
init_w_means = np.zeros((train_X.shape[1], 1))
posterior_covariance = compute_posterior_covariance(train_X, assumed_sigma_sq, init_w_cov)
posterior_means = compute_posterior_mean(train_X, y, assumed_sigma_sq, posterior_covariance, init_w_means, init_w_cov)
mus, sigma_squares = posterior_predictive_ditribution(plot_X, posterior_means, posterior_covariance, assumed_sigma_sq)
selected_mu, selected_sigma_square = mus[selected], sigma_squares[selected]
spread = 5 * selected_mu
lower_lim = selected_mu - spread
upper_lim = selected_mu + spread
ys_to_plot = np.linspace(lower_lim, upper_lim, 100)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()
plot_posterior_predictive(ys_to_plot, selected_mu, selected_sigma_square, ax=ax)
plt.show()


# %%
def plot_posterior_predictive_multivariate(mus, sigmas_squared, ax, num_elems=100):
    sigmas = np.sqrt(sigmas_squared)
    spread = 3 * sigmas
    lower_lims = mus - spread
    upper_lims = mus + spread
    # ys_to_plot = np.linspace(lower_lims, upper_lims, num_elems)[:,:,0].T
    ys_to_plot = np.linspace(lower_lims.squeeze(), upper_lims.squeeze(), num_elems, axis=0).T
    # print(ys_to_plot)
    u = np.repeat(mus, num_elems, axis=1)
    print(u.shape)
    v = ys_to_plot
    # plot_xy_flat = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
    # print(mus.squeeze())
    # print(sigmas.squeeze())
    z = stats.norm.pdf(ys_to_plot, mus, sigmas)
    # z = stats.norm.pdf(mus, , ys_to_plot)


    ax.scatter(u.squeeze(), v.squeeze(), z.squeeze(), antialiased=True, alpha=1)
    ax.set_xlabel('mus')
    ax.set_ylabel('y_new')
    ax.set_zlabel('p(y_new|x_new, X, y, σ²)')
    ax.view_init(10, 45)
    return ax




#     # ax.scatter(u, v, z.reshape(u.shape), cmap=cm.coolwarm, antialiased=False, alpha=1)
#     ax.scatter(u, v, z, cmap=cm.coolwarm, antialiased=False, alpha=1)
#     ax.set_xlabel('x_new')
#     ax.set_ylabel('y_new')
#     ax.set_zlabel('p(y_new|x_new, X, y, σ²)')
#     return ax


assumed_sigma_sq = 500
init_w_cov = np.eye(train_X.shape[1])
init_w_means = np.zeros((train_X.shape[1], 1))
posterior_covariance = compute_posterior_covariance(train_X, assumed_sigma_sq, init_w_cov)
posterior_means = compute_posterior_mean(train_X, y, assumed_sigma_sq, posterior_covariance, init_w_means, init_w_cov)
mus, sigma_squares = posterior_predictive_ditribution(plot_X, posterior_means, posterior_covariance, assumed_sigma_sq)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
plot_posterior_predictive_multivariate(mus, sigma_squares, ax)
plt.show()


# %%
# def plot_posterior_predictive_multivariate(x, mus, sigmas_squared, ax, num_elems=10, with_mid_line=False):
#     # sigmas = np.sqrt(sigmas_squared)
#     sigmas = sigmas_squared
#     spread = 1 * sigmas
#     lower_lims = mus - spread
#     upper_lims = mus + spread
#     x = range(len(mus))
#     if with_mid_line:
#         ax.plot(x, mus)
#     ax.fill_between(x, lower_lims.flatten(), upper_lims.flatten(), color='b', alpha=.1)
#     ax.set_xlabel('data_point')
#     ax.set_ylabel('mus')
def plot_posterior_predictive_multivariate(x, mus, sigmas_squared, ax, num_elems=10, with_mid_line=False):
    # sigmas = np.sqrt(sigmas_squared)
    sigmas = sigmas_squared
    spread = 1 * sigmas
    lower_lims = mus - spread
    upper_lims = mus + spread
    ys_to_plot = np.linspace(lower_lims.squeeze(), upper_lims.squeeze(), num_elems, axis=0).T
    # print(ys_to_plot)
    # u = np.repeat(mus, num_elems, axis=1)
    # v = ys_to_plot
    z = stats.norm.pdf(ys_to_plot, mus, sigmas)

    x = range(len(mus))
    if with_mid_line:
        ax.plot(x, mus)
    ax.fill_between(x, lower_lims.flatten(), upper_lims.flatten(), color='b', alpha=.1)
    ax.set_xlabel('data_point')
    ax.set_ylabel('mus')



assumed_sigma_sq = 50
init_w_cov = np.eye(train_X.shape[1])
init_w_means = np.zeros((train_X.shape[1], 1))
fig = plt.figure(figsize=(15, 5))
selected = 3
plot_X_sorted = plot_X.sort_values(plot_X.columns[selected]) 
ax = fig.add_subplot()
posterior_covariance = compute_posterior_covariance(train_X, assumed_sigma_sq, init_w_cov)
posterior_means = compute_posterior_mean(train_X, y, assumed_sigma_sq, posterior_covariance, init_w_means, init_w_cov)
mus, sigma_squares = posterior_predictive_ditribution(plot_X_sorted, posterior_means, posterior_covariance, assumed_sigma_sq)
xs = plot_X_sorted.iloc[:, selected]
plot_posterior_predictive_multivariate(xs, mus, sigma_squares, ax, with_mid_line=True)


plt.show()
# %%

grid_X = np.random.uniform(-5, 5, (100, X.shape[1]))
plot_X = add_bias_vector(create_polinomial_bases(grid_X, order=p_ord))
# grid_X = np.repeat(np.linspace(-5, 5, 100)[:, None], X.shape[1], 1)

xs = grid_X[:, 0]
ys = grid_X[:, 1]
mu_s = plot_X @ posterior_means
mu_s.shape
# %%

# log(Π p(t|w*x,s)) = ∑ log(p(t|w*x,s))
def compute_likelihood(X, y, w, sigma_square):
    # https://math.stackexchange.com/a/2478240/706135
    mu_s = (X @ w).reshape(-1, 1)
    sigma_square_s = np.repeat(sigma_square, len(mu_s)).reshape(-1, 1)
    sigma_s = np.sqrt(sigma_square_s)
    # print(y.shape, mu_s.shape, sigma_s.shape)
    likelihoods = stats.norm.pdf(y, mu_s, sigma_s)
    # ∑ log(P(t|Xw,sigma))
    sum_log_likelihoods = np.log(likelihoods).sum()
    return sum_log_likelihoods


compute_likelihood(train_X, y, w_pred, sigma_squared_pred)

# %%
# # https://stackoverflow.com/questions/24767355/individual-alpha-values-in-scatter-plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, mu_s, c='red', marker='o')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# %%
def grid_search_run(X, y, w, sigma_squared, n=33, param_num=[]):
    estimates_params = np.hstack([w.T, sigma_squared])
    expanded_params = np.repeat(estimates_params, n, axis=0)
    scale = 100
    random_vals = (np.random.rand(n, expanded_params.shape[1]) - 0.5) * scale
    # N(mu, s^2) + U(-.5, .5) * 10 => N(mu + U(-5, 5), s^2 + U(-5, 5))
    # random_params = np.hstack([expanded_params + random_vals])
    random_params = random_vals
    # random_params = random_vals
    if param_num:
        not_random_param_dim = [i for i in range(estimates_params.shape[1]) if i not in param_num]
        random_params[:, not_random_param_dim] = expanded_params[:, not_random_param_dim]

    random_params = np.vstack([estimates_params, random_params])
    sampled_ws, sampled_sigmas = random_params[:, :-1], random_params[:, -1].reshape(-1, 1)
    all_sums = []
    for w_row, sigma_row in zip(sampled_ws, sampled_sigmas):
        all_sums.append(compute_likelihood(X, y, w_row, sigma_row))
    likelihoods = np.array(all_sums).reshape(-1, 1)
    return np.hstack([sampled_ws, np.sqrt(sampled_sigmas), np.exp(likelihoods)])


param_idxs = [1, 2]
n_samples = 1000
results = grid_search_run(train_X, y, w_pred, sigma_squared_pred, n_samples, param_idxs)
sampled_ws, sampled_likelihoods = results[:, :-1], results[:, -1]
results.shape
# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
sorted_results = results[np.argsort(results[:, -1])]
lbls = [f"w0 {ps[1]:.2f} - w1 {ps[2]:.2f} - likelihood {ps[-1]:.2f}" for ps in sorted_results]

ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Likelihood')
ax.scatter(results[:-1, 1], results[:-1, 2], np.log(results[:-1, -1]))
ax.scatter(results[-1, 1], results[-1, 2], np.log(results[-1, -1]), c='red', s=100, marker='^')
plt.show()
# %%
param_idxs = [3, 4]
results = grid_search_run(train_X, y, w_pred, sigma_squared_pred, n_samples, param_idxs)
sampled_ws, sampled_likelihoods = results[:, :-1], results[:, -1]
results.shape
# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
sorted_results = results[np.argsort(results[:, -1])]

ax.set_xlabel('w3')
ax.set_ylabel('w4')
ax.set_zlabel('Likelihood')
ax.scatter(results[:-1, 3], results[:-1, 4], np.log(results[:-1, -1]))
ax.scatter(results[-1, 3], results[-1, 4], np.log(results[-1, -1]), c='red', s=100, marker='^')
plt.show()
# %%
# Fisher information - Inverse of covariance(w_pred)
# Tells you how much information is in the dataset -- The higher the more information
I_pred = (1 / sigma_squared_pred) * train_X.T @ train_X
I_pred
# %%
# covariance of w_pred
# Tells you how much you need to change a w if you changed a correlated weight
# cov[w_hat|p(t|X,w,σ²)] = E[w_hat²|p(t|X,w,σ²)] - E[w_hat|p(t|X,w,σ²)]²
cov_w_pred = sigma_squared_pred * np.linalg.inv(train_X.T @ train_X)
w_sampled = np.random.multivariate_normal(w_pred.squeeze(), cov_w_pred)
w_sampled

# %%
# Estimate uncertain estimate for the next prediction
#
#
#
# %%
# Next prediction

y_new, sigma_squared_new = next_predictions(val_X[0, None], train_X, w_pred, sigma_squared_pred)
y_new, np.sqrt(sigma_squared_new)
# %%
y_new, sigma_squared_new = next_predictions(val_X, train_X, w_pred, sigma_squared_pred)
pd.DataFrame(np.hstack([y_new, sigma_squared_new, np.sqrt(sigma_squared_new)]), columns="y s^2 s".split())


# %%
def compute_co_distances_faf(X, no_diag=True):
    all_dim_diffs = X[:, None, :] - X
    squared_diffs = all_dim_diffs**2
    sum_of_squares = squared_diffs.sum(axis=-1)
    distances = np.sqrt(sum_of_squares)
    if no_diag:
        h, w = distances.shape
        distances_without_diagonal = distances + np.nan_to_num((np.eye(h, w) * np.inf))
        distances = distances_without_diagonal
    return distances


plot_X = pd.DataFrame(val_X, columns=[f"feat_{ft}" for ft in range(val_X.shape[1])])

# compute_inner_distances_faf(plot_X[["feat_1", "feat_2"]].values)
selected_features = ["feat_1", "feat_2"]
co_distance_matrix = compute_co_distances_faf(plot_X[selected_features].values)
plot_X["distances"] = co_distance_matrix.min(axis=-1)
plot_X["closest_point"] = co_distance_matrix.argmin(axis=-1)

index_closest_point = plot_X.sort_values('distances')["closest_point"].values
plot_X = plot_X.loc[index_closest_point].sort_values(selected_features[0], ascending=True)

plot_X[selected_features]

# %%
# Shows the function of the predicted weights and their uncertainty
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
u = plot_X[selected_features[0]]
v = plot_X[selected_features[1]]
u, v = np.meshgrid(u, v)

plot_X_flat = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
plot_X_polym = add_bias_vector(create_polinomial_bases(plot_X_flat, order=p_ord))
y_new, sigma_squared_new = next_predictions(plot_X_polym, train_X, w_pred, sigma_squared_pred)
ax.scatter(plot_X_polym[:, 1], plot_X_polym[:, 2], y_new.squeeze(), c=np.sqrt(sigma_squared_new).squeeze())
plt.show()
# %%
param_count = len(w_pred) - 2
fig, axes = plt.subplots(param_count, 1, figsize=(45, 15), subplot_kw={"projection": "3d"})
sample_count = 10
selected_features = ["feat_1", "feat_2"]
u = plot_X[selected_features[0]]
v = plot_X[selected_features[1]]
u, v = np.meshgrid(u, v)
plot_X_flat = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
plot_X_polym = add_bias_vector(create_polinomial_bases(plot_X_flat, order=p_ord))

for i, ax in enumerate(axes):
    for j in range(sample_count - 1):
        w_sampled = np.random.multivariate_normal(w_pred.squeeze(), cov_w_pred)

        y_sampled, sigma_squared_new = next_predictions(plot_X_polym, train_X, w_sampled, sigma_squared_pred)
        ax.plot_surface(u, v, y_sampled.reshape(u.shape), antialiased=False, alpha=0.1, cmap='binary')

    y_sampled, sigma_squared_new = next_predictions(plot_X_polym, train_X, w_pred, sigma_squared_pred)
    ax.plot_surface(u, v, y_sampled.reshape(u.shape), antialiased=True, alpha=1)
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    ax.set_zlabel('prediction')
