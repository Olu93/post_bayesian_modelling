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
    # MxD => 1x1 * MxDx1 @ DxD
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
    e = a + 0.5 * (N * M)

    # NxM
    term1 = Y**2
    # NxM = (MxD @ DxN).T
    term2 = (2 * (W @ mu_x.T)).T
    f = (b + 0.5 * (term1 - term2 + last_part).sum())
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
    approx_posteriors = []
    N, M = Y.shape  # N-observed. M-dimensional input vectors y_n
    D = dim
    I_D = np.eye(D)
    exp_tau = a / b
    e, f = a, b
    # MxD
    exp_W = np.random.multivariate_normal(np.zeros(D), I_D, size=M)
    # MxDxD
    exp_WW = I_D + np.einsum('ijk,ikj->ijk', exp_W[:, None], exp_W[:, None])

    for i in range(2):
        sigma_X = compute_sigma_x(N, exp_tau, exp_WW)
        mu_X = compute_mu_x(Y, exp_tau, sigma_X, exp_W)

        exp_X = mu_X
        exp_xx = compute_exp_cov_x(sigma_X, mu_X)

        sigma_W = compute_sigma_w(M, exp_tau, exp_xx)
        mu_W = compute_mu_w(Y, exp_tau, sigma_W, exp_X)

        exp_W = mu_W
        exp_WW = compute_exp_cov_w(sigma_W, mu_W)

        last_part = compute_final_exp(exp_WW, exp_xx, exp_X)
        e, f = compute_exp_tau(e, f, Y, exp_W, exp_X, last_part)
        exp_tau = e / f

        approx_posteriors.append({
            "Q_X": (mu_X, sigma_X),
            "Q_W": (mu_W, sigma_W),
            "Q_T": (e, f),
        })
    return exp_W, pd.DataFrame(exp_X), approx_posteriors


_, low_X, _ = probablistic_principal_component_analysis(normed_X.values, dim=3)

fig = plt.figure(figsize=(15, 7))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122, projection="3d")
_, projected_data, _ = probablistic_principal_component_analysis(
    normed_X.values, dim=3)
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


# %%
def compute_sigma_x_Z(N, exp_tau, exp_cov_w, Z):
    # exp_tau:      1x1
    # exp_cov_w:    MxDxD
    # Z:            NxM
    D = exp_cov_w.shape[1]
    # DxD
    I_D = np.eye(D)

    # NxDxD
    exp_cov_w_Z = np.einsum('nm,mde->nde', Z, exp_cov_w)
    # NxDxD
    new_sigma_x = np.linalg.inv(I_D + exp_tau * (exp_cov_w_Z))

    return new_sigma_x


# Σ_µm
def compute_sigma_w_Z(M, exp_tau, exp_cov_x, Z):
    # exp_tau:      1x1
    # exp_cov_x:    NxDxD
    # Z:            NxM
    D = exp_cov_x.shape[1]
    # DxD
    I_D = np.eye(D)

    # MxDxD
    exp_cov_x_Z = np.einsum('nm,nde->mde', Z, exp_cov_x)

    # MxDxD
    new_sigma_w = np.linalg.inv(I_D + exp_tau * exp_cov_x_Z)

    return new_sigma_w


def compute_mu_x_Z(Y, exp_tau, cov_x_N, exp_w_M, Z):
    # Y:            N x M
    # exp_tau:      1 x 1
    # exp_w_M:      M x D
    # Z:            N x M

    # NxD
    # -- Summation over M is handled by @-multiplication
    YxW = ((Z * Y) @ exp_w_M)
    # NxD => 1x1 * NxDxD * NxDx1
    result = exp_tau * np.einsum('ijk,ij->ik', cov_x_N, YxW)
    return result


def compute_mu_w_Z(Y, exp_tau, cov_w_M, exp_x_N, Z):
    # Y: N x M
    # exp_tau = 1 x 1
    # exp_w_M: N x D
    # cov_x_n: M x D x D

    # MxD
    # -- Summation over M is handled by @-multiplication
    YxX = ((Z.T * Y.T) @ exp_x_N)
    # MxD => 1x1 * DxD @ MxDx1
    result = exp_tau * np.einsum('ijk,ij->ik', cov_w_M, YxX)
    return result


def compute_missing_Yh_params(mu_w_M, mu_x_N, last_part, tau):
    # mu_w_M:       MxD
    # mu_x_N:       NxD
    # last_part:    NxM
    # tau:          1x1

    # NxM
    exp_Yh = (mu_w_M @ mu_x_N.T).T
    # NxM
    exp_Yh_var = last_part + (1 / tau) - (exp_Yh**2)
    return (exp_Yh, exp_Yh_var)


def compute_exp_tau_Z(a, b, Y, W, mu_x, last_part, Z):
    # a:        1x1
    # b:        1x1
    # Y:        NxM
    # W:        MxD
    # mu_x:     NxDx1
    # last_part:NxM
    # Z:        NxM

    # 1x1
    e = a + 0.5 * Z.sum()

    # NxM
    term1 = Y**2
    # NxM = (MxD @ DxN).T
    term2 = (2 * (W @ mu_x.T)).T
    f = (b + 0.5 * (Z * (term1 - term2 + last_part)).sum())
    return e, f


def probablistic_principal_component_analysis_w_missing_values(
    Y,
    dim=3,
    a=1,
    b=1,
    num_iter=10,
):
    approx_posteriors = []
    N, M = Y.shape  # N-observed. M-dimensional input vectors y_n
    Z = np.isreal(Y) * 1
    D = dim
    I_D = np.eye(D)
    exp_tau = a / b
    e, f = a, b

    # MxD
    exp_W = np.random.multivariate_normal(np.zeros(D), I_D, size=M)
    # MxDxD
    exp_WW = I_D + np.einsum('ijk,ikj->ijk', exp_W[:, None], exp_W[:, None])

    for i in range(num_iter):
        sigma_X = compute_sigma_x_Z(N, exp_tau, exp_WW, Z)
        mu_X = compute_mu_x_Z(Y, exp_tau, sigma_X, exp_W, Z)

        exp_X = mu_X
        exp_XX = compute_exp_cov_x(sigma_X, mu_X)

        sigma_W = compute_sigma_w_Z(M, exp_tau, exp_XX, Z)
        mu_W = compute_mu_w_Z(Y, exp_tau, sigma_W, exp_X, Z)

        exp_W = mu_W
        exp_WW = compute_exp_cov_w(sigma_W, mu_W)

        last_part = compute_final_exp(exp_WW, exp_XX, exp_X)
        e, f = compute_exp_tau_Z(e, f, Y, exp_W, exp_X, last_part, Z)
        exp_tau = e / f

        approx_posteriors.append({
            "Q_X": (mu_X, sigma_X),
            "Q_W": (mu_W, sigma_W),
            "Q_T": (e, f),
        })
    return exp_W, pd.DataFrame(exp_X), approx_posteriors


_, low_X, _ = probablistic_principal_component_analysis_w_missing_values(
    normed_X.values, dim=3)

fig = plt.figure(figsize=(15, 7))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122, projection="3d")
_, projected_data, _ = probablistic_principal_component_analysis_w_missing_values(
    normed_X.values,
    dim=3,
)
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


# %%
def compute_sigma_x_lZ(N, exp_tau, exp_cov_w, Z):
    # exp_tau:      1x1
    # exp_cov_w:    MxDxD
    # Z:            NxM
    D = exp_cov_w.shape[1]
    # DxD
    I_D = np.eye(D)

    # NxDxD
    exp_cov_w_Z = np.repeat(exp_cov_w.sum(0)[None], N, axis=0)
    # NxDxD
    new_sigma_x = np.linalg.inv(I_D + exp_tau * (exp_cov_w_Z))

    return new_sigma_x

def compute_exp_y_h(exp_w, exp_x, tau, mu_x, sigma_x):
    pass


def compute_mu_x_lZ(Y, exp_tau, cov_x_N, exp_w_M, Z):
    # Y:            N x M
    # exp_tau:      1 x 1
    # exp_w_M:      M x D
    # Z:            N x M

    exp_Y = None
    Yo = (Z * Y)
    Yh = ((1 - Z) * exp_Y)
    YHO = Yo + Yh
    # NxD
    # -- Summation over M is handled by @-multiplication
    YHOxW = YHO @ exp_w_M
    # NxD => 1x1 * NxDxD * NxDx1
    result = exp_tau * np.einsum('ijk,ij->ik', cov_x_N, YHOxW)
    return result


def probablistic_principal_component_analysis_w_latent_missing_values(
    Y,
    dim=3,
    a=1,
    b=1,
    num_iter=10,
):
    approx_posteriors = []
    N, M = Y.shape  # N-observed. M-dimensional input vectors y_n
    Z = np.isreal(Y) * 1
    D = dim
    I_D = np.eye(D)
    exp_tau = a / b
    e, f = a, b

    # MxD
    exp_W = np.random.multivariate_normal(np.zeros(D), I_D, size=M)
    # MxDxD
    exp_WW = I_D + np.einsum('ijk,ikj->ijk', exp_W[:, None], exp_W[:, None])

    for i in range(num_iter):
        sigma_X = compute_sigma_x_lZ(N, exp_tau, exp_WW, Z)
        # Needs to compute a pre mu_X
        mu_X = compute_mu_x_lZ(Y, exp_tau, sigma_X, exp_W, Z)

        exp_X = mu_X
        exp_XX = compute_exp_cov_x(sigma_X, mu_X)

        sigma_W = compute_sigma_w_Z(M, exp_tau, exp_XX, Z)
        mu_W = compute_mu_w_Z(Y, exp_tau, sigma_W, exp_X, Z)

        exp_W = mu_W
        exp_WW = compute_exp_cov_w(sigma_W, mu_W)

        last_part = compute_final_exp(exp_WW, exp_XX, exp_X)
        e, f = compute_exp_tau_Z(e, f, Y, exp_W, exp_X, last_part, Z)
        exp_tau = e / f

        approx_posteriors.append({
            "Q_X": (mu_X, sigma_X),
            "Q_W": (mu_W, sigma_W),
            "Q_T": (e, f),
        })
    return exp_W, pd.DataFrame(exp_X), approx_posteriors


_, low_X, _ = probablistic_principal_component_analysis_w_latent_missing_values(
    normed_X.values, dim=3)

fig = plt.figure(figsize=(15, 7))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122, projection="3d")
_, projected_data, _ = probablistic_principal_component_analysis_w_latent_missing_values(
    normed_X.values,
    dim=3,
)
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
