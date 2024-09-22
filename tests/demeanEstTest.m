
n = 200;
m = 20;


gamma = 3;
b = [1 2 -0.3 2.5];
delta = 2;



theta_cov = [1 0.5; 0.5 0.8];
theta = mvnrnd(zeros(1,2), theta_cov, n);


x_cov = [1 0.5; 0.5 0.8];



z = abs(theta(:,2)-theta(:,1)) + randn(n,m);



err = mvnrnd(zeros(1,2), x_cov,n * m);
x = reshape(theta, n, 1, 2) + reshape(err, n, m, 2);



y = b(1) + theta(:,1) * b(2) + theta(:,2) * b(3) + theta(:,2).^2 * b(4) + delta * z + randn(n,m) * 1.5;



var(y,0,'all')
var(theta(:,1), 0,'all')
var(theta(:,2), 0,'all')
var(theta(:,2).^2, 0,'all')
var(z, 0,'all')

[y_minus_z, z_coef] = demeanEst(y, z);



theta_test = [ones(n,1) theta(:,1) theta(:,2) theta(:,2).^2];


regress(y_minus_z, theta_test)


x_mean = mean(x, 2);



x_cov_est = cov(reshape(x - x_mean, n*m, 2)) / (m-1)
