# %%
from operator import pos
from typing import Any, Callable, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import meshgrid
import pandas as pd
from IPython.display import display
from scipy import stats
from scipy import special as sps
from scipy.stats import binom
from matplotlib import cm
import random as rnd
import abc
import tqdm

from helper import add_bias_vector, create_polinomial_bases

DiscreteDistribution = Dict[float, float]
# %%
# Follows graphical model of generating r. The param r is now dependent on alpha and beta
true_a = 42
true_b = 13
true_r = true_a / (true_a + true_b)
N_const = 100

ks = np.random.randint(0, 100, size=N_const)
rv = binom(N_const, true_r)
coint_tosses = rv.rvs()
X_obs_tosses = rv.rvs(size=100)
X_obs_llhs = rv.pmf(ks)
display(f"Numer of heads: {coint_tosses}")
display(X_obs_tosses)
display(X_obs_llhs)


# %%
class Predictor(abc.ABC):
    def marginal_likelihood(self, **kwargs):
        pass

    def posterior(self, **kwargs):
        pass

    def likelihood(self, **kwargs):
        pass

    def prior(self, **kwargs):
        pass

    def joint_probability(self, **kwargs):
        pass

    def expected_value(self, **kwargs):
        pass

    def variance(self, **kwargs):
        pass

    def sample(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class DiscretePredictor(Predictor):
    def __init__(self, prior_probs) -> None:
        super().__init__()
        self.prior_probs = prior_probs

    def sample(self, **kwargs):
        size = kwargs.get('num_samples', 1)
        vals = list(self.prior_probs.keys())
        probs = list(self.prior_probs.values())
        sampled = np.random.choice(vals, size=size, p=probs)
        return sampled

    def expected_value(self, **kwargs):
        return np.sum([r * prob for r, prob in self.prior_probs.items()])

    # E_px_square = np.sum([r**2 for r in sampled_r]) / num_samples # WRONG
    # E_px_square = np.sum([self.likelihood(r=r) * r**2 for r in sampled_r]) # WRONG
    # E_px_square = np.sum([self.prior_probs[r] * r**2 for r in sampled_r])  # WRONG
    def variance(self, **kwargs):
        expected_value = self.expected_value()
        # E[P(r|data)^2]
        E_px_square = np.sum([self.prior_probs[r] * r**2 for r in self.prior_probs.keys()])

        # E[P(r|data)]^2
        E_square_px = expected_value**2
        var_x = E_px_square - E_square_px
        return var_x


class ContinuousPredictor(Predictor):
    def __init__(self, precision: int = 100) -> None:
        super().__init__()
        self.precision = 100
        self.candindate_rs = np.linspace(0, 1, self.precision)


## %%
def plot_discrete_dist(predictor: DiscretePredictor,
                       fig: plt.Figure = None,
                       ax: plt.Axes = None,
                       figsize: Tuple = (5, 5)):
    fig = fig or plt.figure(figsize=figsize)
    ax = ax or fig.add_subplot()
    rs, ps = zip(*predictor.prior_probs.items())
    ax.plot(rs, ps)
    ax.set_xlabel("r")
    ax.set_ylabel("P(r|data)")
    exp_val = predictor.expected_value()
    var_val = predictor.variance()
    ax.set_title((f"E[P(r|data)] = {exp_val:.5f}, var[P(r|data)] = {var_val:.5f}"))

    return fig, ax


def plot_continuous_dist(predictor: Predictor,
                         fig: plt.Figure = None,
                         ax: plt.Axes = None,
                         figsize: Tuple = (5, 5),
                         precision=100):
    fig = fig or plt.figure(figsize=figsize)
    ax = ax or fig.add_subplot()
    rs, ps = zip(*predictor.posterior().items())
    ax.plot(rs, ps)
    ax.set_xlabel("r")
    ax.set_ylabel("P(r|data)")
    exp_val = predictor.expected_value()
    var_val = predictor.variance()
    ax.set_title((f"E[P(r|data)] = {exp_val:.5f}, var[P(r|data)] = {var_val:.5f}"))

    return fig, ax


# %%
# Core idea is to oserve y_N which is determined by r, a, b to estimate those params!
class BetaEstimator(ContinuousPredictor):
    def __init__(self, y_N, N, k, precision: int = 100, lr=0.1) -> None:
        super().__init__(precision=precision)
        self.N = N
        self.y_N = y_N
        self.k = k
        self.lr = lr
        self.candindate_rs = np.linspace(0, 1, self.precision + 1)
        self.candindate_as = np.linspace(0, 100, self.precision + 1)
        self.candindate_bs = np.linspace(0, 100, self.precision + 1)

    # p(r,α,β|y_N) -- y_N is OBSERVED!
    def posterior(self, r, alpha, beta):
        pass

    def joint_probability(self, r, alpha, beta):
        return self.likelihood(r, alpha, beta) * self.prior(r, alpha, beta)

    def marginal_likelihood(self, **kwargs):
        pass

    # p(y_N|r)  -- Here we assume that r depends on alpha and beta and those have no additional dependency
    def likelihood(self, r, **kwargs):
        return stats.binom(self.N, r).pmf(self.y_N)

    def prior(self, r, alpha, beta):
        # p(α,β|k) = with k being X, w_a, w_b  
        p_ab = p_a * p_b # Both probability distributions need to be estimated
        
        # p(α|X, w_a) * p(β|X, w_b) as α _|_ β 
        # E[p(α|X, w_a)], E[p(β|X, w_b)]
        alpha, beta = self.a_b_estimate(self.X, self.w_a, self.w_b)
        
        # E[p(r|α,β)] 
        r = self.r_estimate(alpha/(alpha+beta),self.w_r)

        # p(r|α,β)
        p_r_given_ab = stats.beta(alpha, beta).pdf(r)
        return p_r_given_ab * p_ab

    def marginal_likelihood(self, y_K):
        # ∑ ∑ ∑ p(y_N|r) * p(r|α,β) * p(α,β|k) | r,α,β
        pass

    def a_b_estimate(self, X, w_a, w_b):
        a = X @ w_a
        b = X @ w_b
        return a, b

    def r_estimate(self, rs, w_r):
        a = rs @ w_r
        return a

    def estimate_weights(self, y, X, lmb=0):
        w = np.linalg.inv(X.T @ X + lmb * np.eye(X.shape[1])) @ X.T @ y
        return w



    def pred_likelihood(self, y, X, w_r, w_a, w_b):
        train_X = add_bias_vector(create_polinomial_bases(X, 3))  # -- Shape: (num_X, order+1)
        A, B = self.a_b_estimate(train_X, w_a, w_b)  # A = f(X, w_a), B = f(X, w_b) -- Shape: (num_X, 1) & (num_X, 1)

        expected_R = A / (A + B)  # E[R|Alpha,Beta] -- Shape (num_X, 1)
        train_pred_R = add_bias_vector(create_polinomial_bases(expected_R, 3))  # -- Shape: (num_X, order+1)

        y_pred = w_r @ train_pred_R  # y_pred = g(f(X, w_r)) -- Shape: (num_X, 1)
        squared_diff = (y - y_pred)**2  # (y - y_pred)**2
        sum_of_squared_diff = np.sum(squared_diff)  # L = ∑ loss_n | n
        new_w_r, new_w_a, new_w_b = self.optimize(squared_diff, train_pred_R, w_r, w_a, w_b)
        return new_w_r, new_w_a, new_w_b

    # def optimize(self, losses, pred_r, X, w_r, w_a, w_b):

    #     outer_derivative = -2 * losses
    #     inner_derivative_w_r = pred_r
    #     new_w_r = w_r - self.lr * np.mean(outer_derivative * inner_derivative_w_r, axis=0)

    #     d_block_0_a = (X @ w_a) * X
    #     d_block_0_b = (X @ w_b) * X
    #     d_block_1 = -(-(d_block_0_a) / ((X @ w_a + X @ w_b)**2))
    #     inner_derivative_w_a = d_block_1 + d_block_0_a + d_block_0_b
    #     inner_derivative_w_b = d_block_1
    #     new_w_a = w_a - self.lr * np.mean(outer_derivative * w_r * inner_derivative_w_a)
    #     new_w_b = w_b - self.lr * np.mean(outer_derivative * w_r * inner_derivative_w_b)

    #     return new_w_r, new_w_a, new_w_b

    def optimize(self, losses, pred_r, X, w_r, w_a, w_b):

        outer_derivative = -2 * losses
        inner_derivative_w_r = pred_r
        new_w_r = w_r - self.lr * np.mean(outer_derivative * inner_derivative_w_r, axis=0)

        a = X * (X @ w_a)
        b = X * (X @ w_b)
        c = (X @ w_a + X @ w_b)**2
        inner_derivative_w_a = b / c
        inner_derivative_w_b = -a / c
        new_w_a = w_a - self.lr * np.mean(outer_derivative * w_r * inner_derivative_w_a, axis=0)
        new_w_b = w_b - self.lr * np.mean(outer_derivative * w_r * inner_derivative_w_b, axis=0)

        return new_w_r, new_w_a, new_w_b
