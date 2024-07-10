import numpy as np
from NPMLE.utils.data_generators import beta_bernoulli, beta_bernoulli_cheat
from NPMLE.utils.NPMLE_util import *


# Setting Parameters
a = 1.3
b = 1.6
rho = -0.1
epsilon_sigma = 0.15
beta = np.array([0.2, -0.1])
rng = np.random.default_rng(10239324234678)
lg = np.linspace(0.01, 0.99, 40)
grid = make_grid(lg)

beta_bernoulli_kwargs = {'a': a,
                         'b': b,
                         'rho': rho,
                         'epsilon_sigma': epsilon_sigma,
                         'beta': beta,
                         'rng': rng}


NPMLE_kwargs = {'eta': 0.03,
                'gtol': 0.0015,
                'cheat_weight_init': True}


def cheat_pms(x, y):
    return beta_bernoulli_cheat(x, y, a, b)


run_kwargs = {
    'lg': lg,
    'generator': beta_bernoulli,
    'generator_kwargs': beta_bernoulli_kwargs,
    'NPMLE_kwargs': NPMLE_kwargs,
    # 'cheat': cheat_pm
}
