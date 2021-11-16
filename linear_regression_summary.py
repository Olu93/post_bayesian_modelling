# %%
# https://medium.com/@luckecianomelo/the-ultimate-guide-for-linear-regression-theory-918fe1acb380

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
class Predictor(abc.ABC):
    def __init__(self, polym_order, train_data, val_data=None, noise_assumption=1) -> None:
        super().__init__()
        self.polym_order = polym_order
        self.train_X, self.train_y = self.prepare_data(train_data)
        self.val_X, self.val_y = self.prepare_data(val_data)
        self.weights = None
        self.noise_assumption = noise_assumption

    def train(self):
        pass

    def predict(self, data):
        return self.next_predictions(data, self.weights)

    def predict_val(self):
        return self.predict(self.val_X)

    def prepare_data(self, data):
        X = add_bias_vector(create_polinomial_bases(data[:, :-1], order=self.polym_order))
        y = data[:, -1][:, None]
        return X, y

    def plot(self):
        pass

    def compute_posterior_covariance(self, X, sigma_sq, init_w_cov):
        # covariance is the covariance of the normal distribution we assume for the posterior
        posterior_covariance = np.linalg.inv((1 / sigma_sq) * X.T @ X + np.linalg.inv(init_w_cov))
        return posterior_covariance

    def compute_posterior_mean(self, X, y, sigma_sq, posterior_cov, init_w_means, init_w_cov):
        # mean is the mean of the normal distribution we assume for the posterior
        posterior_means = posterior_cov @ ((1 / sigma_sq) * X.T @ y + np.linalg.inv(init_w_cov) @ init_w_means)
        return posterior_means

    def compute_sigma_squared(self, X, y, w):
        # Same as (sum((y_n - x_n*w)²))/N = ((y-Xw).T @ (y-Xw))/N
        total_sigma = y.T @ y - y.T @ X @ w
        expected_sigma = total_sigma / len(y)
        return expected_sigma

    # Next predictions if you have an estimate for w and sigma_squared
    def next_predictions(self, x_new, w_pred):
        y_new = x_new @ w_pred
        X = self.train_X
        y = self.train_y
        expected_sigma_sq = self.compute_sigma_squared(X, y, w_pred)
        cov_w_pred = expected_sigma_sq * np.linalg.inv(X.T @ X)
        sigma_squared_new = ((x_new @ cov_w_pred) * x_new).sum(axis=1)

        return y_new, sigma_squared_new[:, None]


class MLELinearRegression(Predictor):
    def train(self):
        X = self.train_X
        y = train_y
        # I = self.eye(len(y))
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y


class RegularisedLinearRegression(Predictor):
    def __init__(self, polym_order, train_data, val_data=None, lmb=1) -> None:
        super().__init__(polym_order, train_data, val_data=val_data)
        self.lmb = lmb

    def train(self):
        X = self.train_X
        y = train_y
        I = self.eye(len(y))
        init_w_mu = self.zeros(X.shape[1])[:, None]
        err_sigma_sq = self.noise_assumption
        reg_factor = err_sigma_sq / self.lmb
        init_w_cov = reg_factor * I  # Lambda is a combination of base noise and reg term. Hence, noise assumption remains 1
        posterior_cov = I.copy()
        weight_estimate = self.compute_posterior_mean(X, y, 1, posterior_cov, init_w_mu, init_w_cov)
        self.weights = weight_estimate


class MAPLinearRegression(Predictor):
    def __init__(self, polym_order, train_data, val_data=None, sigma_assumption=1) -> None:
        super().__init__(polym_order, train_data, val_data=val_data)

    def train(self):
        X = self.train_X
        y = train_y
        I = self.eye(len(y))
        init_w_mu = self.zeros(X.shape[1])[:, None]
        err_sigma_sq = self.noise_assumption
        reg_factor = err_sigma_sq / self.lmb
        init_w_cov = reg_factor * I  # Lambda is a combination of base noise and reg term. Hence, noise assumption remains 1
        posterior_cov = self.compute_posterior_covariance(X, err_sigma_sq, init_w_cov)
        weight_estimate = self.compute_posterior_mean(X, y, err_sigma_sq, posterior_cov, init_w_mu, init_w_cov)
        self.weights = weight_estimate