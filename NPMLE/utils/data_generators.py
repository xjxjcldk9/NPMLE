import numpy as np
from scipy.stats import norm, beta, multivariate_normal
from src.utils.NPMLE_util import OLS
from scipy.integrate import nquad


def coupled_marginal_pdf(u, rho, pdf, cdf):
    prod = 1/np.sqrt(1-rho**2)
    prod *= pdf(u[0]) * pdf(u[1])
    a = norm.ppf(cdf(u[0]))
    b = norm.ppf(cdf(u[0]))
    prod *= np.exp(-rho * (rho * (a**2+b**2) - 2*a*b) / (2 * (1-rho**2)))
    return prod


def generate_correlated_beta(n, a, b, rho, rng):
    V1 = rng.normal(size=n)
    V2 = rho * V1 + np.sqrt(1-rho**2) * rng.normal(size=n)
    data = np.zeros((n, 2))

    data[:, 0] = beta.ppf(norm.cdf(V1), a=a, b=b)
    data[:, 1] = beta.ppf(norm.cdf(V2), a=a, b=b)
    return data


def beta_pdf_helper(x, a, b):
    return beta.pdf(x, a=a, b=b)


def beta_cdf_helper(x, a, b):
    return beta.cdf(x, a=a, b=b)


def beta_bernoulli(a, b, rho, epsilon_sigma, beta, n, rng):
    theta = generate_correlated_beta(n, a, b, rho, rng)

    x = rng.binomial(1, p=theta)

    epsilon = rng.normal(scale=epsilon_sigma, size=n)
    y = theta @ beta + epsilon

    def likelihood(x, grid):
        arr = np.power(grid, x[:, None, :]) * \
            np.power(1 - grid, 1 - x[:, None, :])
        return arr[..., 0] * arr[..., 1]

    def prior(grid):
        def pdf(x): return beta_pdf_helper(x, a, b)
        def cdf(x): return beta_cdf_helper(x, a, b)
        return np.apply_along_axis(coupled_marginal_pdf, 1, grid, rho, pdf, cdf)
    return theta, x, y, likelihood, prior


def beta_bernoulli_pms_quad(a, b, rho):
    def pdf(x): return beta_pdf_helper(x, a, b)
    def cdf(x): return beta_cdf_helper(x, a, b)

    def prior(t0, t1):
        return coupled_marginal_pdf(np.array([t0, t1]), rho, pdf, cdf)

    def ber(t0, t1, x0, x1):
        return t0**x0 * (1-t0)**(1-x0) * t1**x1 * (1-t1)**(1-x1)

    def objective_denom(t0, t1, x0, x1):
        return ber(t0, t1, x0, x1) * prior(t0, t1)

    def objective_0(t0, t1, x0, x1):
        return t0 * ber(t0, t1, x0, x1) * prior(t0, t1)

    output = np.zeros((4, 2))

    pm0, _ = nquad(objective_0, [[0, 1], [0, 1]], args=(0, 0))
    denom00, _ = nquad(objective_denom, [[0, 1], [0, 1]], args=(0, 0))
    output[0, :] = np.array([pm0, pm0])/denom00

    pm0, _ = nquad(objective_0, [[0, 1], [0, 1]], args=(0, 1))
    denom01, _ = nquad(objective_denom, [[0, 1], [0, 1]], args=(0, 1))

    output[1, 0] = pm0/denom01
    output[2, 1] = output[1, 0]

    pm0, _ = nquad(objective_0, [[0, 1], [0, 1]], args=(1, 0))
    denom10, _ = nquad(objective_denom, [[0, 1], [0, 1]], args=(1, 0))

    output[1, 1] = pm0/denom10
    output[2, 0] = output[1, 1]

    pm0, _ = nquad(objective_0, [[0, 1], [0, 1]], args=(1, 1))
    denom11 = 1-denom00-denom01-denom10

    output[3, :] = np.array([pm0, pm0])/denom11

    return output

#TODO: 優化這個
def beta_bernoulli_cheat(x, y, a, b, rho):
    # first calculate 4 possible pms
    pms = beta_bernoulli_pms_quad(a, b, rho)

    def mapping(x):
        if (x == np.array([0, 0])).all():
            return pms[0, :]
        if (x == np.array([0, 1])).all():
            return pms[1, :]
        if (x == np.array([1, 0])).all():
            return pms[2, :]
        if (x == np.array([1, 1])).all():
            return pms[3, :]

    # some function that maps 4 possible
    pm_x = np.apply_along_axis(mapping, 1, x)

    return OLS(y, pm_x)


def normal_normal(theta_cov, x_cov, epsilon_sigma, beta, n, rng):
    theta = rng.multivariate_normal(mean=[0, 0], cov=theta_cov, size=n)

    x = np.zeros((n, 2))
    for i in range(n):
        x[i, :] = rng.multivariate_normal(theta[i, :], cov=x_cov)

    epsilon = rng.normal(scale=epsilon_sigma, size=n)
    y = theta @ beta + epsilon

    def likelihood(x, grid):
        return multivariate_normal.pdf(x[:, None]-grid, cov=x_cov)

    def prior(grid):
        return multivariate_normal.pdf(grid, cov=theta_cov)
    return theta, x, y, likelihood, prior


def normal_normal_cheat(x, y, theta_cov, x_cov):
    theta_cov_inv = np.linalg.inv(theta_cov)
    x_cov_inv = np.linalg.inv(x_cov)
    denom = np.linalg.inv(theta_cov_inv + x_cov_inv)
    return OLS(y, x @ x_cov_inv @ denom)
