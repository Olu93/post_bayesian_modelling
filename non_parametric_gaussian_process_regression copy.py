# %matplotlib notebook
# %%
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.lib.function_base import meshgrid
from numpy.random import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from matplotlib import cm
import random as r
from data import observed_data, observed_data_linear, observed_data_wobbly, true_function_polynomial
from helper import add_bias_vector, create_polinomial_bases
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from IPython import get_ipython
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tqdm.notebook import tqdm
from viz import animate_3d_fig
import seaborn as sns
from sklearn.linear_model import LinearRegression
# %%
get_ipython().run_line_magic("matplotlib", "inline")
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
# %%
n = 500
variance = 0.1
space = 10


def simple_2d_func(p1=1, p2=1):
    def func(x, y):
        return (((x + y)**p1) + (x * y)**p2)

    return func


def complex_2d_func(p1=1, p2=1):
    def func(x, y):
        return ((np.sin(x) * p1) / y**p2)

    return func


def more_complex_2d_func(p1=1, p2=1):
    def func(x, y):
        return ((x * np.sin(x))) / (np.log((y**2) - y.min() + 0.01))

    return func


def observed_data_3d(d: int = 10, space=10, variance=0, func=None):
    x, y = np.random.uniform(-space, space, size=(2, d))
    z = func(x, y) + np.random.randn(d) * variance
    return x, y, z


true_func = more_complex_2d_func(2, 2)
data = np.vstack(np.array(observed_data_3d(n, space, variance, true_func)).T)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(data[:, 0], data[:, 1], data[:, 2])
# data = pd.DataFrame(data).sort_values([0, 1]).drop_duplicates().values

# %%
train_X, val_X, train_y, val_y = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, shuffle=True)
scaler_X = MinMaxScaler()
scaler_X.fit(data[:, :-1])
scaler_y = MinMaxScaler((-1, 1))
scaler_y.fit(data[:, -1].reshape(-1, 1))

## %%
train_X = scaler_X.transform(train_X)
val_X = scaler_X.transform(val_X)
train_y = scaler_y.transform(train_y[:, None])
val_y = scaler_y.transform(val_y[:, None])

num_features = train_X.shape[1]

train_X.mean()


# %%

# Incomplete!!! -- Almost done
def neural_network_kernel(pi=1, gammas=None):
    def kernel(X1, X2):
        lr = LinearRegression()
        # (N x M+1)
        X1_w_bias = add_bias_vector(X1)
        # (K x M+1)
        X2_w_bias = add_bias_vector(X2)
        # (M+1 x 1)
        lr.fit(X1_w_bias, X2)
        # (M+1 x M+1) @ IDENTITY
        W = lr.coef_ @ np.eye(len(gammas))
        # (N x M+1) = (N x M+1) @ (M+1 x M+1)
        X1W = X1_w_bias @ W
        # (K x M+1) = (K x M+1) @ (M+1 x M+1)
        X2W = X2_w_bias @ W
        # (N x 1) <= (N x K) <= (N x K x M+1) = (N x M+1) @ (K x M+1)
        X1WX2 = np.diag(2 * np.einsum('nm,km->nk', X1W, X2_w_bias))
        # (N x 1) <= (N x N) <= (N x N x M+1) = (N x M+1) @ (N x M+1)
        X1WX1 = np.diag(2 * np.einsum('nm,km->nk', X1W, X1_w_bias))
        # (K x 1) <= (K x K) <= (K x K x M+1) = (K x M+1) @ (K x M+1)
        X2WX2 = np.diag(2 * np.einsum('nm,km->nk', X2W, X2_w_bias))
        # (N x K) <= (N x 1) / (N x K)
        denominator = np.sqrt((1 + X1WX1) @ (1 + X2WX2))
        dist_matrix = X1WX2 / denominator
        # (N x K)
        return (2 / pi) * np.arcsin(dist_matrix)

    return kernel


def gaussian_process(X, y, X_, noise=0.0, kernel=None):
    C = kernel(X, X)
    C_ = kernel(X_, X_)
    R = kernel(X, X_)
    f = y

    inv_C = np.linalg.inv(C + np.square(noise) * np.eye(C.shape[0]))
    mu_pred = R.T @ inv_C @ f
    sigma_pred = C_ - (R.T @ inv_C @ R)
    # sigma_pred = 0

    return mu_pred, sigma_pred


def plot_predictions_2D(val_y, pred_y, pred_cov_y, x, ax, skip, num_samples=10):
    len_x = len(x)
    r_x = x
    std_y = np.sqrt(np.diag(pred_cov_y))

    ax.plot(r_x, val_y.flat, color="red", alpha=1, label="true")
    ax.plot(r_x, pred_y.flat, color="blue", alpha=1, label="pred")
    ax.scatter(r_x[::skip], val_y[::skip].flat, color="red", alpha=1, label="true")
    ax.scatter(r_x[::skip], pred_y[::skip].flat, color="blue", alpha=1, label="pred")
    ax.fill_between(
        r_x,
        pred_y.flatten() - 3 * std_y,
        pred_y.flatten() + 3 * std_y,
        color='blue',
        alpha=.1,
    )

    sample_functions = np.random.multivariate_normal(
        pred_y.flatten(),
        pred_cov_y,
        num_samples,
    )
    for i, sampled_ys in enumerate(sample_functions):
        ax.plot(
            r_x,
            sampled_ys,
            color="grey",
            lw=1,
            ls='--',
            # label='sample_{}'.format(i),
            alpha=0.2)
    # ax.set_xlim(r_x.min()*0.99, r_x.max()*(1.01))
    ax.legend()


tmp_kernel = polynomial_kernel(0.3, 11)
new_val_x = np.repeat(np.linspace(scaler_X.data_min_.min(), scaler_X.data_max_.max(), 100)[None], 2, axis=0).T
# new_val_x[:, 0] = 1
new_val_y = true_func(new_val_x[:, 0], new_val_x[:, 1])[:, None]
tmp_x = scaler_X.transform(new_val_x)
tmp_y = scaler_y.transform(new_val_y)
pred_y, pred_cov_y = gaussian_process(train_X, train_y, tmp_x, noise=0.07, kernel=tmp_kernel)
fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111)
skip = 5
plot_predictions_2D(
    tmp_y,
    pred_y,
    pred_cov_y,
    tmp_x[:, 1],
    ax,
    skip,
    num_samples=20,
)
# %%
tmp_kernel = gaussian_kernel(0.3, 0.01)
pred_y, pred_cov_y = gaussian_process(train_X, train_y, val_X, noise=0.1, kernel=tmp_kernel)

tmp_x = val_X
# tmp_x = pd.DataFrame(val_X).sort_values([1]).values

fig = plt.figure(figsize=(20, 7))

ax = fig.add_subplot(111, projection="3d")
skip = 1


def plot_predictions_3D(val_y, pred_y, pred_cov_y, x, ax, skip):
    # https://stackoverflow.com/questions/54994600/pyplot-legend-poly3dcollection-object-has-no-attribute-edgecolors2d
    def surf3dplot_legend_fix(surfs):
        for surf in surfs:
            surf._facecolors2d = surf._facecolor3d
            # surf.set_facecolor("white")
            # surf._facecolors2d = "white"
            surf._edgecolors2d = surf._edgecolor3d

    len_x = len(x)
    r_x, r_y = x[:, 0], x[:, 1]
    std_y = np.sqrt(np.diag(pred_cov_y))

    surf1 = ax.plot_trisurf(
        r_x,
        r_y,
        val_y.flat,
        color="white",
        # facecolors="white",
        alpha=0.7,
        label="true")
    surf1.set_edgecolor("red")
    # surf1.set_sort_zpos(-1)
    surf2 = ax.plot_trisurf(r_x, r_y, pred_y.flat, color="white", alpha=0.7, label="pred")
    surf2.set_edgecolor("blue")
    surfs = [surf1, surf2]
    surf3dplot_legend_fix(surfs)
    ax.legend()


plot_predictions_3D(val_y, pred_y, pred_cov_y, tmp_x, ax, skip)

# animate_3d_fig(fig, ax, 45, 100, blit=True)
# %%
# LINUX ONLY!!!
from multiprocessing import Pool, pool, cpu_count
from itertools import product

workers = Pool(4)

alphas = np.power(10.0, np.arange(-3, 2))
gammas = np.linspace(0.001, 2, 6)
noises = np.power(10.0, np.arange(-5, 1))
n_repeats = 10
folds = 10
pg = tqdm(total=np.product((n_repeats, 3, len(noises), len(alphas), len(gammas))))
kernel_functions = [linear_kernel, polynomial_kernel, gaussian_kernel]

results_json = []


def execute_single_experiment(args):
    X_train, X_test, y_train, y_test, i, kernel_func, noise, alpha, gamma = args
    kernel = kernel_func(alpha, gamma)
    y_pred, y_pred_cov = gaussian_process(X_train, y_train, X_test, noise=noise, kernel=kernel)
    return {
        "Iteration": i,
        "Kernel": kernel_func.__name__,
        "Noise": noise,
        "Alpha": alpha,
        "Gamma": gamma,
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }


for i, (tr_ids, te_ids) in enumerate(KFold(n_repeats).split(train_X)):
    X_train, X_test = train_X[tr_ids], train_X[te_ids]
    y_train, y_test = train_y[tr_ids], train_y[te_ids]
    params = product([X_train], [X_test], [y_train], [y_test], [i], kernel_functions, noises, alphas, gammas)
    for result in workers.imap_unordered(execute_single_experiment, params, chunksize=200):
        results_json.append(result)
        pg.update(1)

# %%
results = pd.DataFrame(results_json)
mean_results = results.groupby("Noise Kernel Alpha Gamma".split()).mean().drop("Iteration", axis=1)
mean_results = mean_results.reset_index()
mean_results
# %%
gammas_names = [f"{g:.3f}" for g in gammas]
alphas_names = [f"{g:.3f}" for g in alphas]
fig, axes = plt.subplots(len(noises), len(kernel_functions))
fig.set_size_inches((20, 30))
for index, df in mean_results.groupby("Noise Kernel".split()):
    n = list(noises).index(index[0])
    k = list([kn.__name__ for kn in kernel_functions]).index(index[1])
    ax = axes[n, k]
    data = df.pivot(index='Alpha', columns='Gamma', values='MAPE')
    hm = sns.heatmap(data, ax=ax, yticklabels=alphas, xticklabels=gammas_names)
    ax.set_ylabel("Alpha")
    ax.set_xlabel("Gamma")
    ax.set_title(f"{index[1]} w noise {noises[n]}")
fig.tight_layout()
plt.show()

# %%
