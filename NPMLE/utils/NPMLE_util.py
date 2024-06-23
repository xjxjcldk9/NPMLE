import pandas as pd
import seaborn as sns
import ot
import matplotlib.pyplot as plt
import numpy as np


# TODO: Refactor this code
def make_grid(lg):
    '''
    given a line grid (k), output a squared grid (k**2, 2).
    grid[i,0] = lg[i%k], 
    grid[i,1] = lg[i//k]
    from left to right, bottom to top
    '''
    kn = len(lg)
    kn2 = kn**2
    grid = np.zeros((kn2, 2))
    for i in range(kn2):
        grid[i, 0] = lg[i % kn]
        grid[i, 1] = lg[i // kn]

    return grid


def compute_gradient(M, p):
    """
    M: cost array (n by kn2)
    p: current probabilities (kn2)
    Output: update gradient, loss at p
    """
    n = M.shape[0]
    a = np.ones(n) / n
    b = p
    loss, log = ot.sinkhorn2(a, b, M, reg=1, method="sinkhorn", log=True)

    res = log['v']
    return res - res.mean(), loss


def NPMLE(x, likelihood, grid, eta=0.1, gtol=1e-5, max_iter=1000, verbose=False, cheat_weight_init=None, weight_init=None):
    """
    x: (n, 2) 
    grid: (kn2, 2)
    eta: learning rate
    Output: weights for each grid point
    """
    # 1. Initialization
    kn2 = grid.shape[0]

    if cheat_weight_init:
        w = weight_init(grid)
    else:
        w = np.random.uniform(size=kn2)
    w /= w.sum()

    # 2. Iterate
    M = - np.log(likelihood(x, grid))

    gnorm = np.inf
    i = 0
    while (gnorm > gtol) and (i < max_iter):
        u, loss = compute_gradient(M, w)
        w -= eta * u
        gnorm = np.linalg.norm(u)
        if verbose:
            print(f"iteration {i}: loss={loss:.4f}, gnorm={gnorm:.4f}")
        i += 1
    return w


def OLS(y, X):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def split_weight(w):
    '''
    input w(kn2). 
    Output(kn, kn).
    output[i, j] = w[kn*j+i]
    left to right, then bottom to top
    '''
    kn2 = w.shape[0]
    kn = int(np.sqrt(kn2))
    output = np.zeros((kn, kn))
    for i in range(kn):
        for j in range(kn):
            output[i, j] = w[kn*j+i]
    return output


def plot_heat_prior(w, w_grid_prior, grid, plot_save, n):
    '''
    plot w on grids. Also plot true theta and x on the same plot
    '''
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html#sphx-glr-gallery-images-contours-and-fields-interpolation-methods-py
    # grid 的以及繪圖設計

    kn = int(np.sqrt(grid.shape[0]))
    lower = grid[0][0]
    upper = grid[-1][-1]

    ws = [w, w_grid_prior]
    titles = ['Estimated Theta Prior', 'True Theta Prior']

    for i in range(2):
        im = axs[i].imshow(split_weight(ws[i]),  origin='lower',
                           interpolation='nearest')
        fig.colorbar(im, label='Probability')

        ticks = np.arange(kn)
        labels = np.round(np.linspace(lower, upper, kn), 2)
        axs[i].set_xticks(ticks, labels, rotation=90)
        axs[i].set_yticks(ticks, labels, rotation=0)
        axs[i].set_title(titles[i])

    fig.suptitle(f'n={n:,}')
    plt.savefig(plot_save)


def normalize_axis1(arr):
    '''normalize at axis=1'''
    return arr / arr.sum(axis=1, keepdims=True)


def calculate_post_mean(x, w, likelihood, grid):
    post_joint = likelihood(x, grid) * w
    return normalize_axis1(post_joint) @ grid


def proposed_run(lg, generator, NPMLE_beta_return=False, True_prior_grid_beta_return=False, True_post_mean_beta_return=False,
                 post_mean_func=None, plot=False, plot_save=None, generator_kwargs={}, NPMLE_kwargs={}):
    '''
    This function output run NPMLE routine and return second stage beta estimation using
    NPMLE and true prior grid.

    If post_mean_func is given, it uses the true posterior mean to run the second stage OLS instead of 
    the discrete calculation.

    post_mean_func only accept (x, y)

    Can specify which kind of beta you want to return
    '''
    grid = make_grid(lg)
    theta, x, y, likelihood, prior = generator(**generator_kwargs)

    if NPMLE_kwargs.get('cheat_weight_init'):
        NPMLE_kwargs['weight_init'] = prior

    return_betas = {}

    if NPMLE_beta_return:
        w = NPMLE(x, likelihood, grid, **NPMLE_kwargs)

    w_grid_prior = prior(grid)
    w_grid_prior /= w_grid_prior.sum()

    if plot:

        plot_heat_prior(w, w_grid_prior, grid,
                        plot_save, generator_kwargs['n'])
        return

    if NPMLE_beta_return:
        NPMLE_post_mean = calculate_post_mean(x, w, likelihood, grid)
        NPMLE_beta = OLS(y, NPMLE_post_mean)
        return_betas['NPMLE'] = NPMLE_beta

    # TODO: 太慢了！
    if True_prior_grid_beta_return:
        true_prior_grid_post_mean = calculate_post_mean(
            x, w_grid_prior, likelihood, grid)
        true_prior_grid_beta = OLS(y, true_prior_grid_post_mean)
        return_betas['TPrior'] = true_prior_grid_beta

    if True_post_mean_beta_return:
        return_betas['TPost'] = post_mean_func(x, y)

    return return_betas


def simulate_betas(B, run, seed=208903, verbose=False, run_kwargs={}):

    if 'rng' in run_kwargs.keys():
        raise ValueError('Do not specify rng, Use seed!')

    beta_lists = []

    rng = np.random.default_rng(seed)
    run_kwargs['generator_kwargs']['rng'] = rng
    n = run_kwargs['generator_kwargs']['n']

    for b in range(B):
        if verbose and b % 10 == 0:
            print(f'n={n:,}, b={b}')

        betas = run(**run_kwargs)
        for name, beta in betas.items():
            beta_lists.append((beta[0], beta[1], n, name))

    return pd.DataFrame(beta_lists, columns=['0', '1', 'n', 'case'])


def make_betas_consistent_plots(beta_df, beta, save_at, hash_num):
    fig, axs = plt.subplots(2, 2, figsize=(14, 6))
    for i in range(2):
        sns.kdeplot(x=beta_df[str(i)], hue=beta_df['n'], ax=axs[0][i])
        axs[0][i].axvline(beta[i], color='r')
        axs[0][i].set_title(f'beta_{i}')
        axs[0][i].set_xlabel('')

    def grouping_MSE(group):
        MSE = pd.Series()
        for i in range(2):
            MSE[i] = ((group[str(i)] - beta[i])**2).mean()
        return MSE

    MSE_table = beta_df.groupby('n', as_index=False).apply(grouping_MSE)

    for i in range(2):
        axs[1][i].plot(MSE_table['n'], MSE_table[i])
        axs[1][i].set_ylabel('MSE')
        axs[1][i].set_xticks(MSE_table['n'])
        axs[1][i].ticklabel_format(
            axis='y', scilimits=(-3, 6), useMathText=True)

    case = beta_df['case'].iloc[0]

    title = f'{case} distribution'
    fig.suptitle(title)

    plt.savefig(f'{save_at}/{hash_num}_{case}')
