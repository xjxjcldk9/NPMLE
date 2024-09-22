oldpath = path;
path(oldpath, '../matlab_codes');


n = 200;
m = 20;

prior = 'uniform';
gamma = 3;
beta = [1; 2; -0.3; 2.5];
delta = [0.4; 0.5];

x_cov = [1 0.5; 0.5 0.8];


[y, x, z, theta] = data_generator(n, m, prior, x_cov, gamma, delta, beta);

size(theta)
size(y)
size(x)
size(z)