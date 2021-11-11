# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import meshgrid
import pandas as pd
from IPython.display import display
from scipy import stats
from matplotlib import cm
import random as r
from data import observed_data, observed_data_linear, true_function_polynomial
from helper import add_bias_vector, create_polinomial_bases
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
# %%
n = 50
val_n = 10
p_ord = 2
data = np.vstack(np.array(observed_data(n)).T)
val_data = np.vstack(np.array(observed_data(val_n)).T)
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
# t_n = Xw + ε   |ε ~ N(0,1) 
# Model assumes p(t|X,w,σ²) = N(Xw; σ²I)


def compute_weights(X, y):
    # Under assumptions unbiased: w = E[w_hat|p(t|X,w,σ²)] = Σ[w_hat*p(t|X,w,σ²)] over t
    # w_hat
    w_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    return w_hat


def compute_sigma_squared(X, y, w): 
    # Same as (sum((y_n - x_n*w)²))/N = ((y-Xw).T @ (y-Xw))/N
    total_sigma = y.T @ y - y.T @ X @ w
    expected_sigma = total_sigma / len(y)
    return expected_sigma


w_pred = compute_weights(train_X, y).reshape(-1, 1)
sigma_squared_pred = compute_sigma_squared(train_X, y, w_pred)

w_pred, np.sqrt(sigma_squared_pred)
# %%

grid_X = np.random.uniform(-5, 5, (100, X.shape[1]))
# grid_X = np.repeat(np.linspace(-5, 5, 100)[:, None], X.shape[1], 1)
plot_X = add_bias_vector(create_polinomial_bases(grid_X, order=p_ord))

xs = grid_X[:, 0]
ys = grid_X[:, 1]
mu_s = plot_X @ w_pred
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
def next_predictions(x_new, X, w_pred, sigma_squared_pred):
    t_new = x_new @ w_pred
    cov_w_pred = sigma_squared_pred * np.linalg.inv(X.T @ X)
    sigma_squared_new = ((x_new @ cov_w_pred) * x_new).sum(axis=1)

    return t_new, sigma_squared_new[:, None]


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
