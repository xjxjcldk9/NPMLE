import numpy as np
from NPMLE.utils.ALM import DualALM
from scipy.stats import norm as normal


seed = 232423
rng = np.random.default_rng(seed)


n = 1000
G_support = [0, 3]
proportion = 0.5
theta = rng.choice(G_support, size=n, p=[1-proportion, proportion])
z = rng.normal(size=n)
Y = theta + z


m = 500
mu = np.linspace(np.min(Y),np.max(Y),m)


L = normal.pdf(Y[:,None]-mu)


DualALM(L)