# %%
# Follows: Rogers, Simon, and Mark Girolami. A First Course in Machine Learning, Second Edition. 2nd ed. Chapman & Hall/CRC, 2016. https://nl1lib.org/book/11571143/9a0735.
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
from laplace_approximation_with_taylor_expansion import Y
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


# %%
# Σ_xn
def compute_sigma_x(N, exp_tau, exp_cov_w):
    # exp_tau:      1x1
    # exp_cov_w:    MxDxD
    D = exp_cov_w.shape[1]
    # DxD
    I_D = np.eye(D)

    # DxD
    new_sigma_x = np.linalg.inv(I_D + exp_tau * exp_cov_w.sum(axis=(0)))
    # NxDxD
    all_sigmas = np.repeat(new_sigma_x[None], N, axis=0)
    # Out:          NxDxD
    return all_sigmas


# Σ_µm
def compute_sigma_w(M, exp_tau, exp_cov_x):
    # exp_tau:      1x1
    # exp_cov_x:    NxDxD
    D = exp_cov_x.shape[1]
    # DxD
    I_D = np.eye(D)

    # DxD
    new_sigma_w = np.linalg.inv(I_D + exp_tau * exp_cov_x.sum(axis=0))

    # MxDxD
    all_sigmas = np.repeat(new_sigma_w[None], M, axis=0)
    # Out:          MxDxD
    return all_sigmas


def compute_mu_x(Y, exp_tau, cov_x_N, exp_w_M):
    # Y: N x M
    # exp_tau = 1 x 1
    # exp_w_M: M x D
    # sigma_x_N: N x D x D

    # NxD
    # -- Summation over M is handled by @-multiplication
    YxW = (Y @ exp_w_M)
    # NxD => 1x1 * NxDxD * NxDx1
    result = exp_tau * np.einsum('ijk,ij->ik', cov_x_N, YxW)
    return result


def compute_mu_w(Y, exp_tau, cov_w_M, exp_x_N):
    # Y: N x M
    # exp_tau = 1 x 1
    # exp_w_M: N x D
    # cov_x_n: M x D x D

    # MxD
    # -- Summation over M is handled by @-multiplication
    YxX = (Y.T @ exp_x_N)
    # MxD => 1x1 * DxD @ MxDx1
    result = exp_tau * np.einsum('ijk,ij->ik', cov_w_M, YxX)
    return result


# <xn,xn.T> = Σ_xn + µ_xn @ µ_xn.T
def compute_exp_cov_x(cov_x_N, mu_x):
    # cov_x:    NxDxD
    # mu_x:     NxD

    # NxDxD = NxDx1 @ Nx1xD
    cov_mu_x_N = np.einsum('ijk,ikj->ijk', mu_x[:, None], mu_x[:, None])
    # NxDxD = NxDxD + NxDxD
    return cov_x_N + cov_mu_x_N


def compute_exp_cov_w(cov_w_M, mu_w):
    # cov_w:    MxDxD
    # mu_w:     MxD

    # NxDxD = NxDx1 @ Nx1xD
    cov_mu_x_N = np.einsum('ijk,ikj->ijk', mu_w[:, None], mu_w[:, None])
    # NxDxD = NxDxD + NxDxD
    return cov_w_M + cov_mu_x_N


def compute_exp_tau(a, b, Y, W, mu_x, last_part):
    # a:        1x1
    # b:        1x1
    # Y:        NxM
    # W:        MxD
    # mu_x:     NxDx1
    # last_part:NxM

    # 1x1
    N, M = Y.shape
    e = a + ((N * M) / 2)

    # NxM
    term1 = 0.5 * np.sum((Y**2))
    # MxN = MxD @ DxN
    term2 = (2 * (W @ mu_x.T)).sum()
    f = (b + term1 - term2 + last_part.sum())
    return e, f


def compute_final_exp(cov_ww, sigma_x, mu_x):
    # cov_ww:   MxDxD
    # sigma_x:  NxDxD
    # mu_x:     NxD

    # NxMxDxD
    innter_trace = np.einsum('ijk,hkj->hijk', cov_ww, sigma_x)
    # NxMx1
    trace = np.trace(innter_trace, axis1=-2, axis2=-1)
    # NxMxDxD = Nx1xDx1 @ 1xMxDxD @ Nx1x1xD
    # summation = np.transpose(mu_x, axes=(
    #     0, 2, 1))[:, None, :, :] @ cov_w[None, :, :, :] @ mu_x[:, None, :, :]
    # NxMx1 = Nx1x1xD @ 1xMxDxD @ Nx1x1xD
    mux_x_ww = np.einsum('ij,hjk->ihk', mu_x, cov_ww)
    mux_x_ww_x_mux = np.einsum('ihj,kj->ih', mux_x_ww, mu_x)
    result = trace + mux_x_ww_x_mux
    return result


def probablistic_principal_component_analysis(Y, dim=3, a=1, b=1):
    N, M = Y.shape  # N-observed. M-dimensional input vectors y_n
    D = dim
    I_D = np.eye(D)
    exp_tau = a / b
    e, f = a, b
    # MxD
    exp_w = np.random.multivariate_normal(np.zeros(D), I_D, size=M)
    # MxDxD
    exp_ww = I_D + np.einsum('ijk,ikj->ijk', exp_w[:, None], exp_w[:, None])

    for i in range(2):
        sigma_x = compute_sigma_x(N, exp_tau, exp_ww)
        mu_x = compute_mu_x(Y, exp_tau, sigma_x, exp_w)

        exp_x = mu_x
        exp_xx = compute_exp_cov_x(sigma_x, mu_x)

        sigma_w = compute_sigma_w(M, exp_tau, exp_xx)
        mu_w = compute_mu_w(Y, exp_tau, sigma_w, exp_x)

        exp_w = mu_w
        exp_ww = compute_exp_cov_w(sigma_w, mu_w)

        last_part = compute_final_exp(exp_ww, exp_xx, exp_x)
        e, f = compute_exp_tau(e, f, Y, exp_w, exp_x, last_part)
        exp_tau = e / f

    return exp_w, pd.DataFrame(exp_x)


_, low_X = probablistic_principal_component_analysis(normed_X.values, dim=3)

fig = plt.figure(figsize=(15, 7))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122, projection="3d")
_, projected_data = probablistic_principal_component_analysis(normed_X.values,
                                                              dim=3)
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
