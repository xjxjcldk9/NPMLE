import numpy as np
from NPMLE.parameters.parameters import parameters
from scipy.stats import multivariate_normal


class Discrete_Normal(parameters):
    def __init__(self, rng=2398400678123,
                 lg=np.linspace(-1.5, 1.5, 30),
                 epsilon_sigma=0.06,
                 beta=np.array([0.2, -0.1]), 
                 theta_supoort=np.array([[0.2,0.3],[-0.4,0.1],[-0.15,-0.2],[0.1,-0.3],[0,0]]),
                 theta_prob=np.array([0.2,0.1,0.2,0.1,0.4]),
                 x_cov=np.array([[1, 0], [0, 1]])):
        super().__init__(rng, lg, beta, epsilon_sigma)
        self.theta_supoort = theta_supoort
        self.theta_prob = theta_prob
        self.x_cov = x_cov
        
        self.likelihood = lambda x, g: multivariate_normal.pdf(x[:,None]-g, cov=self.x_cov)
        
    def generate(self, n, B=1):
        for b in range(B):
            theta = self.rng.choice(self.theta_supoort, size=n, p=self.theta_prob)

            X = np.zeros((n, 2))
            for i in range(n):
                X[i, :] = self.rng.multivariate_normal(theta[i, :], cov=self.x_cov)

            epsilon = self.rng.normal(scale=self.epsilon_sigma, size=n)
            y = theta @ self.beta + epsilon
            
            yield theta, X, y
    
    
                
        
    
    


NPMLE_kwargs = {'eta': 0.03,
                'gtol': 0.0015,
                'cheat_weight_init': True}
