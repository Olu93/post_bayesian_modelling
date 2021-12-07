# http://krasserm.github.io/2020/11/04/gaussian-processes-classification/
# %matplotlib notebook
# %%
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.function_base import meshgrid
from numpy.random import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from matplotlib import cm
import random as r
from data import observed_data, observed_data_binary, observed_data_linear, observed_data_wobbly, true_function_polynomial
from helper import add_bias_vector, create_polinomial_bases, sigmoid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from IPython import get_ipython
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score, accuracy_score
from tqdm.notebook import tqdm
from viz import animate_3d_fig
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
# %%
get_ipython().run_line_magic("matplotlib", "inline")
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
# %%
n = 500
variance = 0.1
space = 10
n_iter = 30

data = np.vstack(np.array(observed_data_binary(n)).T)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(data[:, 0], data[:, 1], data[:, 2])
# data = pd.DataFrame(data).sort_values([0, 1]).drop_duplicates().values

# %%
train_X, val_X, train_y, val_y = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, shuffle=True)
scaler_X = StandardScaler()
scaler_X.fit(data[:, :-1])
scaler_y = MinMaxScaler((0, 1))
scaler_y.fit(data[:, -1].reshape(-1, 1))

## %%
# train_X = add_bias_vector(scaler_X.transform(train_X))
train_X = scaler_X.transform(train_X)
val_X = scaler_X.transform(val_X)
train_y = train_y[:, None]
val_y = val_y[:, None]

num_features = train_X.shape[1]

train_X.mean()


# %%
def linear_kernel(alpha=1, gamma=1):
    def kernel(X1, X2):
        gamma = 1
        dist_matrix = np.einsum('nj,jk->nk', X1, X2.T)
        # dist_matrix = np.sum(np.square(X1), axis=1).reshape(-1, 1) + np.sum(np.square(X2), axis=1) - 2 * np.dot(X1, X2.T)
        return alpha * (1 + dist_matrix)**gamma

    return kernel


def minkovsky_kernel(p=2, gamma=None):
    def kernel(X1, X2):
        sq_diffs = (X1 - X2[:, None])**p
        sum_diffs = sq_diffs.sum(axis=-1)
        return (sum_diffs**(1 / p))

    return kernel


def covariance_kernel(alpha=None, gamma=None):
    def kernel(X1, X2):
        X1_centralised = (X1 - X1.mean(axis=1)[:, None])
        X2_centralised = (X2 - X2.mean(axis=1)[:, None])
        cov = (X1_centralised @ X2_centralised.T) / 2
        return cov

    return kernel


def gaussian_kernel(alpha, gamma):
    def kernel(X1, X2):
        diffs = X1[:, None] - X2
        sq_diffs = np.square(diffs)

        dist_matrix = sq_diffs.sum(axis=-1)
        # dist_matrix = np.sum(np.square(X1), axis=1).reshape(-1, 1) + np.sum(np.square(X2), axis=1) - 2 * np.dot(X1, X2.T)
        return (np.square(alpha) * np.exp((-1 / (2 * gamma)) * dist_matrix))

    return kernel


def compute_q(t, f):
    return t - sigmoid(f)


def compute_P(f):
    return (sigmoid(f) * (1 - sigmoid(f))) * np.eye(len(f))


def compute_C(X, kernel_function=linear_kernel()):
    return kernel_function(X, X)


def first_derivation(q, C_inv, f):
    # q: Binary classification comparisons
    # C: Covariance Matrix
    # f: Param subject to Optimization
    return q - (C_inv @ f)


def second_derivation(P, C_inv):
    return -P - C_inv


def compute_posterior_f_MAP(X, X_star, f_pred, kernel):
    # R:        Cov(X , X*) Covariance between train X and test X
    R = kernel(X, X_star)
    # C:        Correlation matrix of seen X | ALREADY inverted!!!
    C = kernel(X, X)
    # C_star:   Cov(X*, X*) Covariance matrix of X from data to predict
    C_star = kernel(X_star, X_star)
    # f:        Estimated f mean
    f = f_pred

    C_inv = np.linalg.inv(C)
    mu_f_star = R.T @ C_inv @ f
    sigma_f_star = C_star - (R.T @ C_inv @ R)
    return mu_f_star, sigma_f_star


def gaussian_process_binary_classification_MAP(X_val, y_val, X_train, y_train, kernel, max_iter=20):
    def predict_y(mu_f):
        return sigmoid(mu_f) > 0.5

    N, F = X_train.shape
    f = np.random.normal(size=(N, 1))

    # Needs to be computed only once!!!
    C = kernel(X_train, X_train)
    C_inv = np.linalg.inv(C).T
    num_true_run_length = max_iter + 1
    all_accs = np.zeros(num_true_run_length)
    all_gradients = np.zeros((num_true_run_length, N, 1))
    all_accs[0] = accuracy_score(y_train, sigmoid(f) > 0.5)
    all_gradients[0] = f
    pbar = tqdm(range(1, num_true_run_length))

    for i in pbar:
        P = compute_P(f)
        q = compute_q(y_train, f)
        df = first_derivation(q, C_inv, f)
        dff = second_derivation(P, C_inv)
        gradient = (np.linalg.inv(dff) @ df)
        f = f - gradient
        mu_f_val, sigma_f_val = compute_posterior_f_MAP(X_train, X_val, f, kernel)
        y_pred = predict_y(mu_f_val)
        all_accs[i] = accuracy_score(y_val, y_pred)
        all_gradients[i] = gradient
        pbar.set_description_str(f"Acc: {all_accs[i]:.2f}")

    return y_pred, mu_f_val, sigma_f_val, (all_accs, all_gradients)


# k_func = covariance_kernel()
# k_func = minkovsky_kernel(2, 0.1)
# k_func = linear_kernel(0.001, 0.1)
k_func = gaussian_kernel(1, 0.01)
preds_MAP, mu_f_val_MAP, sigma_f_val_MAP, info_MAP = gaussian_process_binary_classification_MAP(
    val_X,
    val_y,
    train_X,
    train_y,
    k_func,
    max_iter=n_iter,
)

plt.plot(info_MAP[0])
plt.show()


# %%
def gaussian_process_binary_class_sampling(X_val, y_val, X_train, y_train, kernel, max_iter=20, num_samples=100):
    def predict_y(mu_f, sigma_f):
        f_star_samples = np.random.multivariate_normal(mu_f.flatten(), sigma_f, size=S)
        f_star_S = sigmoid(f_star_samples)
        f_mean_star_S = (f_star_S.sum(axis=0) / S)[:, None]
        return f_mean_star_S > 0.5

    N, F = X_train.shape
    f = np.random.normal(size=(N, 1))

    # C = compute_C(X_train, kernel_function=kernel)  # Needs to be computed only once!!!
    C = kernel(X_train, X_train)
    C_inv = np.linalg.inv(C).T
    R = kernel(X_train, X_val)
    C_star = kernel(X_val, X_val)

    num_true_run_length = max_iter + 1
    all_accs = np.zeros(num_true_run_length)
    all_gradients = np.zeros((num_true_run_length, N, 1))
    all_accs[0] = accuracy_score(y_train, sigmoid(f) > 0.5)
    all_gradients[0] = f
    pbar = tqdm(range(1, num_true_run_length))

    for i in pbar:
        P = compute_P(f)
        q = compute_q(y_train, f)
        df = first_derivation(q, C_inv, f)
        dff = second_derivation(P, C_inv)
        gradient = (np.linalg.inv(dff) @ df)
        f = f - gradient
        # Sampling
        mu_f_val, sigma_f_val = compute_posterior_f_MAP(X_train, X_val, f, kernel)
        y_pred = predict_y(mu_f_val, sigma_f_val)

        all_accs[i] = accuracy_score(y_val, y_pred)
        all_gradients[i] = gradient

        pbar.set_description_str(f"Acc: {all_accs[i]:.2f}")
    return y_pred, mu_f_val, sigma_f_val, (all_accs, all_gradients)


n_iter = 30
S = 42
k_func = gaussian_kernel(1, 0.01)
preds_Sampling, mu_f_val_Sampling, sigma_f_val_Sampling, info_Sampling = gaussian_process_binary_class_sampling(
    val_X,
    val_y,
    train_X,
    train_y,
    k_func,
    max_iter=n_iter,
    num_samples=S,
)

plt.plot(info_Sampling[0])
plt.show()


# %%
def compute_posterior_f_Laplace(X, X_star, mu_f, sigma_f, kernel):
    # R:        Cov(X , X*) Covariance between train X and test X
    R = kernel(X, X_star)
    # C:        Correlation matrix of seen X | ALREADY inverted!!!
    C = kernel(X, X)
    # C_star:   Cov(X*, X*) Covariance matrix of X from data to predict
    C_star = kernel(X_star, X_star)
    # f:        Estimated f mean<

    C_inv = np.linalg.inv(C)
    f_mean_star = R.T @ C_inv @ mu_f
    f_sigma_star = C_star - R.T @ C_inv @ (np.eye(sigma_f.shape[0]) + sigma_f @ C_inv) @ R
    return f_mean_star, f_sigma_star


def gaussian_process_binary_class_laplace(X_val, y_val, X_train, y_train, kernel, max_iter=20, num_samples=100):
    def predict_y(mu_f):
        return sigmoid(mu_f) > 0.5

    N, F = X_train.shape
    mode_f = np.random.normal(size=(N, 1))

    # C = compute_C(X_train, kernel_function=kernel)  # Needs to be computed only once!!!
    C = kernel(X_train, X_train)
    C_inv = np.linalg.inv(C).T
    R = kernel(X_train, X_val)
    C_star = kernel(X_val, X_val)

    num_true_run_length = max_iter + 1
    all_accs = np.zeros(num_true_run_length)
    all_gradients = np.zeros((num_true_run_length, N, 1))
    all_accs[0] = accuracy_score(y_train, sigmoid(mode_f) > 0.5)
    pbar = tqdm(range(1, num_true_run_length))

    for i in pbar:
        P = compute_P(mode_f)
        q = compute_q(y_train, mode_f)
        df = first_derivation(q, C_inv, mode_f)
        dff = second_derivation(P, C_inv)
        gradient = (np.linalg.inv(dff) @ df)
        mode_f = mode_f - gradient
        hessian = second_derivation(compute_P(mode_f), C_inv)
        mu_f_val, sigma_f_val = compute_posterior_f_Laplace(X_train, X_val, mode_f, hessian, kernel)

        y_pred = predict_y(mu_f_val)
        all_accs[i] = accuracy_score(y_val, y_pred)
        all_gradients[i] = gradient

        pbar.set_description_str(f"Acc: {all_accs[i]:.2f}")
    return y_pred, mu_f_val, sigma_f_val, (all_accs, all_gradients)


n_iter = 30
S = 42
k_func = gaussian_kernel(1, 0.01)
preds_Laplace, mu_f_val_Laplace, sigma_f_val_Laplace, info_Laplace = gaussian_process_binary_class_laplace(
    val_X,
    val_y,
    train_X,
    train_y,
    k_func,
    max_iter=n_iter,
    num_samples=S,
)

plt.plot(info_Laplace[0])
plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(val_X[:, 0], val_X[:, 1], mu_f_val_Laplace, c=preds_Laplace)
animate_3d_fig(fig, ax)