# %%
from sklearn import datasets
import abc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
from viz import plot_clustering, plot_contour_2d, plot_countours_and_points, plot_train_val_curve, plot_w_samples
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics
from sklearn import model_selection
from tqdm.notebook import tqdm

# %%
data = datasets.load_wine(return_X_y=True, as_frame=True)
data_X, data_y = data[0], data[1]
data_X
# %%
normed_X = (data_X - data_X.mean()) / data_X.std()
normed_X

# def compute_x_mu(Y, exp_tau, cov_x_N, exp_w_M):
#     # Y: N x M
#     # exp_w_M: D x 1
#     # exp_tau = 1 x 1
#     # exp_w_M: M x D
#     # cov_x_n: D x D

#     # NxD => Σ[NxMx1 @ 1x1xD|M]  
#     # -- Summation over M is handled by @-multiplication
#     y_nm_times_wm = (Y @ exp_w_M.T[None, :, :]).sum(axis=1)
#     # NxDxD => 1x1 * DxD @ NxDx1
#     result = exp_tau * cov_x_N @ y_nm_times_wm[:, :, None]
#     return result 
# def compute_w_mu(Y, exp_tau, cov_w_M, exp_x_N):
#     # Y: N x M
#     # exp_w_M: D x 1
#     # exp_tau = 1 x 1
#     # exp_w_M: M x D
#     # cov_x_n: D x D

#     # NxD => Σ[NxMx1 @ 1x1xD|M]  
#     # -- Summation over M is handled by @-multiplication
#     y_nm_times_wm = (Y @ exp_x_N.T[None, :, :]).sum(axis=1)
#     # NxDxD => 1x1 * DxD @ NxDx1
#     result = exp_tau * cov_w_M @ y_nm_times_wm[:, :, None]
#     return result 


# def compute_x_cov(exp_tau, w_M):
#     # exp_tau = 1 x 1
#     # w_M = M x D
#     D = w_M.shape[1]
#     # D x D
#     I_D = np.eye(D)

#     # DxD = DxD + 1x1 + DxD := Σ[MxDx1 @ Mx1xD|M]
#     cov_x_n = I_D + exp_tau * (w_M[:, :, None] @ w_M[:, None, :]).sum(axis=0)
    
#     # 1xDxD
#     return np.linalg.inv(cov_x_n)[None, :,:]

# def compute_w_cov(exp_tau, x_N):
#     # exp_tau = 1 x 1
#     # x_N = N x D
#     D = x_N.shape[1]
#     # D x D
#     I_D = np.eye(D)

#     # DxD = DxD + 1x1 + DxD := Σ[MxDx1 @ Mx1xD|M]
#     cov_x_n = I_D + exp_tau * (x_N[:, :, None] @ x_N[:, None, :]).sum(axis=0)
    
#     # 1xDxD
#     return np.linalg.inv(cov_x_n)[None, :,:]
# %%
def compute_mu_x(X):
    # X: NxD
    # Dx1
    return X.T.mean(axus=-1)[:, None] 

def compute_mu_w(W):
    # W: MxD
    # Dx1
    return W.T.mean(axus=-1)[:, None] 

def compute_cov_x(cov_x, mu_x):
    # cov_x:    DxD
    # mu_x:     Dx1

    # DxD
    return cov_x + mu_x @ mu_x.T

def compute_cov_w(cov_w, mu_w):
    # cov_x:    DxD
    # mu_x:     Dx1
    
    # DxD
    return cov_w + mu_w @ mu_w.T

def compute_exp_tau(a, b, N, M, Y, mu_wn, mu_x, last_part):
    e = a + ((N*M)/2)
    f = b + 0.5*np.sum((Y**2) - (2 * (mu_wn) @ mu_x) + last_part)
    return e/f

def compute_final_exp(cov_w, cov_x, mu_x):
    trace = np.diag(cov_w@cov_x)
    result = trace + (mu_x.T @ cov_w @ mu_x)
    return result


def probablistic_principal_component_analysis(Y, dim=3, a=1, b=1):
    N, M = Y.shape  # N-observed. M-dimensional input vectors y_n
    D = dim
    I_D = np.eye(D)
    expected_tau = a / b
    # W - MxD
    w_M = stats.multivariate_normal(np.zeros(M)[:, None], I_D)
    # I_D_m + <w_m> @ <w_m>.T
    expected_ww = I_D + w_M @ w_M.T
    expected_x_n = expected_tau

    C = (Y.values.T @ Y.values) / N
    eigvals, eigvecs = np.linalg.eig(C)
    n_highest_eigvecs = eigvecs[:, :dim]
    projection = Y @ n_highest_eigvecs
    return projection


# %%
fig = plt.figure(figsize=(15, 7))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122, projection="3d")
projected_data = probablistic_principal_component_analysis(normed_X, dim=3)
projected_data = projected_data.join(data_y)

for cls in data_y.unique():
    members = projected_data.iloc[:, 3] == cls
    c_projected_data = projected_data.loc[members]
    ax1.scatter(
        x=c_projected_data.iloc[:, 0],
        y=c_projected_data.iloc[:, 1],
        # c=c_projected_data.iloc[:, 3],
        label=f"Class {cls}",
    )
    ax1.legend()
    ax2.scatter(
        xs=c_projected_data.iloc[:, 0],
        ys=c_projected_data.iloc[:, 1],
        zs=c_projected_data.iloc[:, 2],
        # c=c_projected_data.iloc[:, 3],
        label=f"Class {cls}",
    )
    ax2.legend()
    ax2.view_init(30, 45)
plt.show()
Axes3D