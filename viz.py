import numpy as np
from scipy import stats
from matplotlib.patches import Ellipse


def plot_w_samples(all_w_hats,
                   ax,
                   w_cov,
                   w_mu,
                   burn_in_period,
                   title="",
                   precision=2,
                   num_samples=100):
    w_sample_mean_all = all_w_hats.mean(axis=0)
    # w_sample_var_all = np.cov(all_w_hats.T)
    w_sample_mean_relevant = np.mean(all_w_hats[burn_in_period:], axis=0)
    w_sample_cov_relevant = np.cov(all_w_hats[burn_in_period:].T)
    W_Sampled = stats.multivariate_normal(
        w_sample_mean_relevant, w_sample_cov_relevant).rvs(num_samples)
    x_min, y_min = W_Sampled.min(axis=0)
    x_max, y_max = W_Sampled.max(axis=0)
    x_cov = precision * np.sqrt(w_cov[0, 0])
    y_cov = precision * np.sqrt(w_cov[1, 1])
    x_mu = w_mu[0]
    y_mu = w_mu[1]
    x_lims = np.min([x_min, x_mu - x_cov]), np.max([x_max, x_mu + x_cov])
    y_lims = np.min([y_min, y_mu - y_cov]), np.max([y_max, y_mu + y_cov])
    X = np.linspace(x_lims[0], x_lims[1], 100)
    Y = np.linspace(y_lims[0], y_lims[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z_True = stats.multivariate_normal(w_mu, w_cov).pdf(
        np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

    CS = ax.contour(X, Y, Z_True)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.scatter(W_Sampled[:, -2],
               W_Sampled[:, -1],
               s=10,
               c='grey',
               label="Samples from relevant Centroid")
    ax.scatter(w_sample_mean_all[-2],
               w_sample_mean_all[-1],
               s=100,
               marker="s",
               c='red',
               label="Centroid w discarded")
    ax.scatter(w_sample_mean_relevant[-2],
               w_sample_mean_relevant[-1],
               s=100,
               marker="p",
               c='orange',
               label="Centroid w/o discarded")

    # # https://stackoverflow.com/a/18218468
    # # https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    # # https://stackoverflow.com/a/20127387
    lambda_, v = np.linalg.eig(w_sample_cov_relevant)
    lambda_ = np.sqrt(lambda_)
    ellipsis = Ellipse(
        xy=(w_sample_mean_relevant[-2], w_sample_mean_relevant[-1]),
        width=2 * lambda_[0] * np.sqrt(5.991),
        height=2 * lambda_[1] * np.sqrt(5.991),
        angle=np.rad2deg(np.arctan2(*v[:, 0][::-1])),
        edgecolor='r',
        fc='None',
        lw=2,
    )
    ax.add_patch(ellipsis)

    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_title(f"Weight Samples: {title}")
    ax.legend()


def plot_w_path_from_burnin(all_w_hats,
                            ax,
                            w_cov,
                            w_mu,
                            burn_in_period,
                            title="",
                            precision=2):
    all_w_hats_to_use = all_w_hats[burn_in_period:]
    x_min, y_min = all_w_hats_to_use.min(axis=0)
    x_max, y_max = all_w_hats_to_use.max(axis=0)
    x_cov = precision * np.sqrt(w_cov[0, 0])
    y_cov = precision * np.sqrt(w_cov[1, 1])
    x_mu = w_mu[0]
    y_mu = w_mu[1]
    x_lims = np.min([x_min, x_mu - x_cov]), np.max([x_max, x_mu + x_cov])
    y_lims = np.min([y_min, y_mu - y_cov]), np.max([y_max, y_mu + y_cov])
    X = np.linspace(x_lims[0], x_lims[1], 100)
    Y = np.linspace(y_lims[0], y_lims[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z_True = stats.multivariate_normal(w_mu, w_cov).pdf(
        np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

    CS = ax.contour(X, Y, Z_True)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.plot(all_w_hats_to_use[:, -2], all_w_hats_to_use[:, -1])
    ax.scatter(all_w_hats_to_use[:, -2],
               all_w_hats_to_use[:, -1],
               s=10,
               c="blue",
               label="step")
    ax.scatter(all_w_hats_to_use[0][-2],
               all_w_hats_to_use[0][-1],
               s=100,
               c='green',
               label="start")
    ax.scatter(all_w_hats_to_use[-1][-2],
               all_w_hats_to_use[-1][-1],
               s=100,
               c='red',
               label="end")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_title(f"Weight Movement: {title}")
    ax.legend()


def plot_w_path_until_burnin(all_w_hats,
                             ax,
                             w_cov,
                             w_mu,
                             burn_in_period,
                             title="",
                             precision=2):
    all_w_hats = all_w_hats[:burn_in_period]
    x_min, y_min = all_w_hats.min(axis=0)
    x_max, y_max = all_w_hats.max(axis=0)
    x_cov = precision * np.sqrt(w_cov[0, 0])
    y_cov = precision * np.sqrt(w_cov[1, 1])
    x_mu = w_mu[0]
    y_mu = w_mu[1]
    x_lims = np.min([x_min, x_mu - x_cov]), np.max([x_max, x_mu + x_cov])
    y_lims = np.min([y_min, y_mu - y_cov]), np.max([y_max, y_mu + y_cov])
    X = np.linspace(x_lims[0], x_lims[1], 100)
    Y = np.linspace(y_lims[0], y_lims[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z_True = stats.multivariate_normal(w_mu, w_cov).pdf(
        np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

    CS = ax.contour(X, Y, Z_True)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.plot(all_w_hats[:, -2], all_w_hats[:, -1])
    ax.scatter(all_w_hats[:, -2],
               all_w_hats[:, -1],
               s=10,
               c="blue",
               label="step")
    ax.scatter(all_w_hats[0][-2],
               all_w_hats[0][-1],
               s=100,
               c='green',
               label="start")
    ax.scatter(all_w_hats[-1][-2],
               all_w_hats[-1][-1],
               s=100,
               c='red',
               label="end")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.set_title(f"Weight Movement: {title}")
    ax.legend()


def plot_train_val_curve(smooth, all_train_values, all_val_values, ax, y_label):
    ax.plot(all_train_values[::smooth], label=f"train-loss")
    ax.plot(all_val_values[::smooth], label=f"val-loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    # ax.set_title("Accuracies per iteration")
    ax.legend()