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
from data import observed_data_classification, observed_data_classification_two_features
from helper import add_bias_vector, compute_metrics, create_polinomial_bases, predict, sigmoid
from viz import plot_countours_and_points, plot_train_val_curve, plot_w_samples
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics
from sklearn import model_selection
from tqdm.notebook import tqdm
# %%
np.set_printoptions(linewidth=100, formatter=dict(float=lambda x: "%.3g" % x))
IS_EXACT_FORMULA = True

# %%
n = 1000
n_c = 3
x1mu, x2mu = 1, 1
x1std, x2std = 3, 3
xystd = 1
val_n = 100
noise = True
seed = 42
cov_data = np.array([[x1std, xystd], [xystd, x2std]])
mean_data = np.array([x1mu, x2mu])

true_X, true_y, X_means, X_covs = observed_data_classification_two_features(
    n, mean_data, cov_data)

# %%
train_X, val_X, train_y, val_y = train_test_split(true_X,
                                                  true_y,
                                                  test_size=0.2)

ax = plt.gca()
plot_countours_and_points(ax, train_X, train_y, X_means, X_covs)
plt.show()


# %%
def linear_kernel(X_sq_differences):
    pass

def knn_linear_kernel(k, X_new, train_X, train_y, kernel_func=None):
    X_new = X_new[:, None, :]
    X_differences = X_new - train_X
    X_sq_differences = X_differences**2
    X_sum_sq_differences = X_sq_differences.sum(axis=-1)
    X_root_sum_sq_differences = np.sqrt(X_sum_sq_differences)
    X_with_proximity_indices = X_root_sum_sq_differences.argsort(axis=-1)
    X_k_closest = X_with_proximity_indices[:, :k]
    label_of_k_closest = train_y[X_k_closest]
    df_label_of_k_closest = pd.DataFrame(label_of_k_closest.squeeze())
    voting_winners = df_label_of_k_closest.apply(lambda x: x.mode(),
                                                 axis=1).iloc[:, 0].values
    return voting_winners


pred_y = knn_linear_kernel(3, val_X, train_X, train_y)
metrics.accuracy_score(val_y, pred_y)
# %%
k_range = range(1, 20)
num_experiments = len(k_range)
kf = model_selection.KFold(n_splits=num_experiments)
train_acc = np.zeros(num_experiments)
val_acc = np.zeros(num_experiments)

for k, (train_idxs, val_idxs) in tqdm(zip(k_range, kf.split(true_X, true_y)), total=num_experiments):
    X_train_subset, X_val_subset = true_X[train_idxs], true_X[val_idxs]
    y_train_subset, y_val_subset = true_y[train_idxs], true_y[val_idxs]
    y_train_pred = knn_linear_kernel(k, X_train_subset, X_train_subset, y_train_subset)
    y_val_pred = knn_linear_kernel(k, X_val_subset, X_train_subset, y_train_subset)
    train_acc[k - 1] = metrics.accuracy_score(y_train_subset, y_train_pred)
    val_acc[k - 1] = metrics.accuracy_score(y_val_subset, y_val_pred)

plt.plot(k_range,train_acc, label="train")
plt.plot(k_range,val_acc, label="val")
plt.legend()
