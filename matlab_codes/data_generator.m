function [y, x, z, theta] = data_generator(n, m, prior, x_cov, gamma, delta, beta)


%弄成3個不同長相的prior。1. uniform, 2. normal, 3. 長相奇特的

switch prior
    case 'uniform'
        rho = 0.3;
        z = normrnd(0,1,n,2);
        z(:,2) = rho * z(:,1) + sqrt(1-rho^2) * z(:,2);
        theta = normcdf(z);
    case 'normal'
        theta_cov = [1 0.5; 0.5 0.8];
        theta = mvnrnd(zeros(1,2),theta_cov,n);
    case 'beta'
        rho = 0.3;
        z = normrnd(0,1,n,2);
        z(:,2) = rho * z(:,1) + sqrt(1-rho^2) * z(:,2);
        theta = betainv(normcdf(z), 0.5, 0.5);
    otherwise
        warning('不支援的prior')
end

    
% theta: (n,2)

% 生成z：(n,m,2)，與theta有關


z = abs(theta(:,2)-theta(:,1)) + randn(n,m,2);


err = mvnrnd(zeros(1,2), x_cov, n * m);
x = reshape(theta, n, 1, 2) + reshape(gamma * z, n, m, 2) + reshape(err, n, m, 2);


theta_matrix = [ones(n, 1) theta theta(:,2).^2];


y = theta_matrix * beta + reshape(reshape(z, n*m,2) * delta, n,m) +  randn(n,m) * 1.5;

end 