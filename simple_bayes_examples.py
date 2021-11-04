# %%
from operator import pos
from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import meshgrid
import pandas as pd
from IPython.display import display
from scipy import stats
from scipy.stats import binom
from matplotlib import cm
import random as rnd
import abc
# %%
true_r = 0.78
N = 100
rv = binom(N, true_r)
# binom.rvs(1, true_r, size=1000) # For individual tosses
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
    
    # Prior P(θ)
    def prior(self, **kwargs):
        r = kwargs.get('r')
        return self.prior_probs.get(r, 0)

    # Prior P(data) = ∑ P(data|θ) * P(θ) | θ
    def marginal_likelihood(self, **kwargs):
        return np.sum([self.likelihood(r=r) * self.prior(r=r) for r in self.prior_probs.keys()])

    # Posterior P(θ|data)
    def posterior(self, **kwargs):
        norm_constant = self.marginal_likelihood()
        return {r: self.joint_probability(r=r) / norm_constant for r in self.prior_probs.keys()}


    def sample(self, **kwargs):
        size = kwargs.get('num_samples', 1)
        vals = list(self.prior_probs.keys())
        probs = list(self.prior_probs.values())
        sampled = np.random.choice(vals, size=size, p=probs)
        return sampled

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
        return super().prior(**kwargs)

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

    def expected_value(self, **kwargs):
        return np.sum([r * prob for r, prob in self.prior_probs.items()])

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

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
faxes = axes.flatten()
N = 50
one_data_sample = binom(N, true_r)

for i, ax in enumerate(faxes):
    fig, ax = plot_discrete_dist(predictor, fig=fig, ax=ax)
    num_heads = one_data_sample.rvs()
    predictor = predictor.update(N=N, num_heads=num_heads)
fig.tight_layout()
plt.show()
# %%
predictor.sample(num_samples=10)


# %%
class GenericPredictorUniformPrior(DiscretePredictor):
    def __init__(self, prior_probs, generator:Callable) -> None:
        super().__init__(prior_probs)
        self.generator = generator or (lambda x: rnd.uniform(0, 1))

    def marginal_likelihood(self, **kwargs):
        pass

    def posterior(self, **kwargs):
        pass

    def likelihood(self, **kwargs):
        llh = self.generator(**kwargs)
        return llh

    def prior(self, **kwargs):
        return super().prior(**kwargs)

    def joint_probability(self, **kwargs):
        pass

    def expected_value(self, **kwargs):
        pass

    def variance(self, **kwargs):
        pass

    def sample(self, **kwargs):
        pass


# %%
class BinomPredictorBetaPrior(Predictor):
    pass


# %%
