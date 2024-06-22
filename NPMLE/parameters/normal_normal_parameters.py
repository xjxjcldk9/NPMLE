import numpy as np
from NPMLE.utils.data_generators import normal_normal, normal_normal_cheat
from NPMLE.utils.NPMLE_util import *

# Setting Parameters
theta_cov = np.array([[0.5, 0.2], [0.2, 0.5]])
x_cov = np.array([[0.7, -0.1], [-0.1, 0.5]])
epsilon_sigma = 0.06
beta = np.array([0.2, -0.1])
rng = np.random.default_rng(2398400678123)
lg = np.linspace(-1.5, 1.5, 30)
grid = make_grid(lg)

normal_normal_kwargs = {'theta_cov': theta_cov,
                        'x_cov': x_cov,
                        'epsilon_sigma': epsilon_sigma,
                        'beta': beta,
                        'rng': rng}


NPMLE_kwargs = {'eta': 0.03,
                'gtol': 0.0015,
                'cheat_weight_init': True}


def cheat_pm(x, y):
    return normal_normal_cheat(x, y, theta_cov, x_cov)


run_kwargs = {
    'lg': lg,
    'generator': normal_normal,
    'generator_kwargs': normal_normal_kwargs,
    'NPMLE_kwargs': NPMLE_kwargs,
    'post_mean_func': cheat_pm
}
