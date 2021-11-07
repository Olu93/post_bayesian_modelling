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

DiscreteDistribution = Dict[float, float]
# %%
true_r = 0.78
N = 100
rv = binom(N, true_r)
coint_tosses = rv.rvs()
f"Numer of heads: {coint_tosses}"


# %%
class Predictor(abc.ABC):
    def update(self, **kwargs):
        pass

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
class BinomPredictorDiscreteUniformPrior(DiscretePredictor):
    def __init__(self, N, num_heads, prior_probs) -> None:
        super().__init__(prior_probs)
        self.N = N
        self.num_heads = num_heads

    # Likelihood P(data|r)
    def likelihood(self, **kwargs):
        r = kwargs.get('r')
        return stats.binom.pmf(self.num_heads, self.N, r)

    # Prior P(r)
    def prior(self, **kwargs):
        r = kwargs.get('r')
        return self.prior_probs.get(r, 0)

    # Prior P(data) = ∑_r P(data|r) * P(r)
    def marginal_likelihood(self, **kwargs):
        return np.sum([self.likelihood(r=r) * self.prior(r=r) for r in self.prior_probs.keys()])

    # Posterior P(r|data)
    def posterior(self, **kwargs):
        norm_constant = self.marginal_likelihood()
        return {r: self.joint_probability(r=r) / norm_constant for r in self.prior_probs.keys()}

    # Posterior P(data|r) * P(r)
    def joint_probability(self, **kwargs):
        r = kwargs.get('r')
        return self.likelihood(r=r) * self.prior(r=r)

    def update(self, **kwargs):
        N = kwargs.get('N')
        num_heads = kwargs.get('num_heads')
        posterior_probs = self.posterior()
        return BinomPredictorDiscreteUniformPrior(N, num_heads, posterior_probs)


# %%
discretisation_steps = 100
predictor = BinomPredictorDiscreteUniformPrior(
    10, 5, {possible_r: 1 / discretisation_steps
            for possible_r in np.linspace(0, 1, discretisation_steps + 1)})

fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharey=True)
faxes = axes.flatten()
N = 50
data_sampler = binom(N, true_r)

for i, ax in enumerate(faxes):
    fig, ax = plot_discrete_dist(predictor, fig=fig, ax=ax)
    num_heads = data_sampler.rvs()
    predictor = predictor.update(N=N, num_heads=num_heads)
fig.tight_layout()
plt.show()
# %%
predictor.sample(num_samples=10)


# %%
class GenericPredictorUniformPrior(DiscretePredictor):
    def __init__(self, params: Dict, generator: Callable, prior_probs: DiscreteDistribution) -> None:
        super().__init__(prior_probs)
        self.params = params
        self.generator = generator
        self.likelihood_function = self.generator(params) or (lambda: rnd.uniform(0, 1))

    # Likelihood P(data|θ)
    def likelihood(self, theta):
        return self.likelihood_function(theta)

    # Prior P(θ)
    def prior(self, theta):
        # print(self.prior_probs.get(theta, 0))
        return self.prior_probs.get(theta, 0)

    # Prior [denominator] P(data) = ∑ P(data|θ) * P(θ) | θ
    def marginal_likelihood(self, **kwargs):
        return np.sum([self.likelihood(theta) * self.prior(theta) for theta in self.prior_probs.keys()])

    # Posterior P(θ|data)
    def posterior(self, **kwargs):
        norm_constant = self.marginal_likelihood()
        return {theta: self.joint_probability(theta) / norm_constant for theta in self.prior_probs.keys()}

    # Joint Probaility [numerator] P(data|θ=x) * P(θ=x)
    def joint_probability(self, theta):
        return self.likelihood(theta) * self.prior(theta)

    def expected_value(self, **kwargs):
        return np.sum([r * prob for r, prob in self.prior_probs.items()])

    def update(self, kwargs):
        posterior_probs = self.posterior()
        return GenericPredictorUniformPrior(kwargs, self.generator, posterior_probs)


discretisation_steps = 100
start_dist = {possible_r: 1 / discretisation_steps for possible_r in np.linspace(0, 1, discretisation_steps + 1)}


# We can use a distance measure between observations and sampled probs to compute a likelihood
# We can do this because the likelihood does not need to be normalized (Energy-Function)
def gen_energy_from_tosses(data, sample_num=1000):
    num_heads = data.get('num_heads')
    N = data.get('N')
    observed_estimate = num_heads / N

    def compute_likelihood(*args):
        theta = args[0]
        sampled_tosses = binom.rvs(1, theta, size=sample_num)  # Individual tosses
        estimate_probality = sampled_tosses.mean()
        # These measures are numerically unstable
        # llh = 1 / (np.abs(observed_estimate - estimate_probality) + 1*np.finfo(np.float128).tiny)  #
        # llh = 1 / (np.abs(observed_estimate - estimate_probality) + 0.00000001)  #
        # Estimation based on difference between observed and sampled
        # llh = 1 - np.abs(observed_estimate - estimate_probality)
        # llh = np.log(np.exp(observed_estimate))-np.log(np.exp(estimate_probality))
        llh = np.exp(1) / np.exp(np.abs(observed_estimate - estimate_probality))
        return llh

    return compute_likelihood


N = 100
data_sampler = binom(N, true_r)
num_heads = data_sampler.rvs()
predictor = GenericPredictorUniformPrior({'num_heads': num_heads, 'N': N}, gen_energy_from_tosses, start_dist)
all_predictors = [predictor]
fig, axes = plt.subplots(3, 3, figsize=(20, 15), sharey=True)
faxes = axes.flatten()
for i, ax in enumerate(tqdm.tqdm(faxes)):
    fig, ax = plot_discrete_dist(predictor, fig=fig, ax=ax)
    num_heads = data_sampler.rvs()
    predictor = predictor.update({'num_heads': num_heads, 'N': N})
    all_predictors.append(predictor)
fig.tight_layout()

plt.show()


# %%
def gen_likelihood_from_tosses(data, sample_num=1000):
    num_heads = data.get('num_heads')
    N = data.get('N')
    observed_estimate = num_heads / N

    def compute_likelihood(*args):
        theta = args[0]
        sampled_tosses = binom.rvs(1, theta, size=sample_num)  # Individual tosses
        estimate_probality = sampled_tosses.mean()
        # Estimation based on difference between observed and sampled
        llh = 1 - (np.abs(observed_estimate - estimate_probality) / np.max([observed_estimate, estimate_probality]))
        return llh

    return compute_likelihood


N = 100
data_sampler = binom(N, true_r)
num_heads = data_sampler.rvs()
predictor = GenericPredictorUniformPrior({'num_heads': num_heads, 'N': N}, gen_likelihood_from_tosses, start_dist)
all_predictors = [predictor]
fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharey=True)
faxes = axes.flatten()
for i, ax in enumerate(tqdm.tqdm(faxes)):
    fig, ax = plot_discrete_dist(predictor, fig=fig, ax=ax)
    num_heads = data_sampler.rvs()
    predictor = predictor.update({'num_heads': num_heads, 'N': N})
    all_predictors.append(predictor)
fig.tight_layout()
plt.show()


# %%
class BinomPredictorBetaPrior(ContinuousPredictor):
    def __init__(self, alpha, beta, N, k, precision=100) -> None:
        super().__init__(precision)
        self.alpha = alpha
        self.beta = beta
        self.N = N
        self.k = k

    # P(data) = ∑ P(data|θ) * P(θ) | θ
    def marginal_likelihood(self, **kwargs):
        return (sps.binom(N, num_heads) * sps.exp(
            sps.gammaln(self.alpha + self.beta) - sps.gammaln(self.alpha) - sps.gammaln(self.beta) +
            sps.gammaln(self.alpha + num_heads) + sps.gammaln(self.beta + N - num_heads) -
            sps.gammaln(self.alpha + self.beta + N)))

    # P(r|k,N)
    def posterior(self, rs=[]):
        rs = rs if any(rs) else self.candindate_rs
        probs = {row[0]: row[1] for row in np.vstack([rs, stats.beta(self.alpha, self.beta).pdf(rs)]).T}
        return probs

    # P(k,N|r)
    def likelihood(self, rs=[]):
        rs = rs if any(rs) else self.candindate_rs
        return stats.binom(self.k, self.N).pmf(rs)

    # P(r)
    def prior(self, rs=[]):
        rs = rs if any(rs) else self.candindate_rs
        return stats.beta(self.alpha - self.k, self.beta - self.N + self.k).pdf(rs)

    def joint_probability(self, rs=[]):
        # r = kwargs.get('r')
        rs = rs if any(rs) else self.candindate_rs
        return self.likelihood(rs=rs) * self.prior(rs=rs)

    def expected_value(self):
        return self.alpha / (self.alpha + self.beta)

    def variance(self):
        # mu = self.expected_value()
        # return (mu * (1 - mu)) / (1 + self.alpha + self.beta)
        return (self.alpha * self.beta) / ((self.alpha + self.beta)**2 + (self.alpha + self.beta + 1))

    def update(self, N, k):
        new_alpha = self.alpha + k
        new_beta = self.beta + N - k
        # print(new_alpha, new_beta)
        return BinomPredictorBetaPrior(new_alpha, new_beta, N, k)

    def sample(self, **kwargs):
        uniform_draw = np.random.uniform()
        sample = stats.beta.ppf(uniform_draw, self.alpha, self.beta)
        return sample

# %%
N = 100
num_heads = data_sampler.rvs()
predictor = BinomPredictorBetaPrior(1, 1, N, num_heads)
all_predictors = [predictor]

fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=True)
faxes = axes.flatten()
for i, ax in enumerate(tqdm.tqdm(faxes)):
    fig, ax = plot_continuous_dist(predictor, fig=fig, ax=ax)
    num_heads = data_sampler.rvs()
    predictor = predictor.update(N, num_heads)
    all_predictors.append(predictor)
fig.tight_layout()
plt.show()

# %%
# Toss for Toss
N = 1
num_heads = 0
predictor = BinomPredictorBetaPrior(1, 1, N, num_heads)
all_predictors = [predictor]

fig, axes = plt.subplots(4, 4, figsize=(15, 15), sharey=True)
faxes = axes.flatten()
for i, ax in enumerate(tqdm.tqdm(faxes)):
    fig, ax = plot_continuous_dist(predictor, fig=fig, ax=ax)
    N += 1 
    num_heads += np.random.uniform() < true_r
    predictor.alpha = 1
    predictor.beta = 1
    predictor = predictor.update(N, num_heads)
    all_predictors.append(predictor)
fig.tight_layout()
plt.show()
# %%
