import numpy as np
from NPMLE.utils.NPMLE_util import compute_gradient, make_grid, split_weight
import matplotlib.pyplot as plt


class parameters:
    def __init__(self, rng, lg, beta, epsilon_sigma):
        self.NPMLE_kwargs = None
        self.run_kwargs = None 
        self.rng = np.random.default_rng(rng)
        self.lg = lg
        self.grid = make_grid(lg)
        self.epsilon_sigma = epsilon_sigma
        self.beta = beta
        self.likelihood = None
    
    
    def plot_heat_prior(self, w):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        
        x = self.grid[:,0]
        y = self.grid[:,1]
        
        
        ax.bar3d(x, y, 0, 0.02, 0.02, w)
    
    
    
    
    
    
    
    def NPMLE(self, X, likelihood='actual', theta=None, eta=0.1, gtol=1e-6, max_iter=1000, verbose=False):
        """
        x: (n, 2) 
        eta: learning rate
        Output: weights for each grid point
        """
        # 1. Initialization
        kn2 = self.grid.shape[0]

        w = np.random.uniform(size=kn2)
        w /= w.sum()

        # 2. Iterate
        if likelihood == 'actual':
            M = - np.log(self.likelihood(X, self.grid))
        else:
            pass
        gnorm = np.inf
        i = 0
        while (gnorm > gtol) and (i < max_iter):
            u, loss = compute_gradient(M, w)
            w -= eta * u
            gnorm = np.linalg.norm(u)
            if verbose:
                if i%50==0:
                    print(f"iteration {i}: loss={loss:.4f}, gnorm={gnorm:.7f}")
            i += 1
        return w
    
    def calculate_post_mean(self, X, w):
        L = self.likelihood(X, self.grid)
        return ((L * w) @ self.grid) / (L @ w)[:, None]

        
    
        
    