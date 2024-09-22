function [b_est, grid, weight, theta, y, pm] = dim2_twostep(n, m, prior, x_cov, gamma, delta, b)




%%%%%%%%%%%%%%%%%%


%估x_cov

x_mean = mean(x, 2);
x_cov_est = cov(reshape(x - x_mean, n*m, 2)) / double(m-1);




[y_minus_z, z_coef] = demeanEst(y, z);

%var(y_minus_z)
%var(theta(:,1))
%var(theta(:,2))
%var(theta(:,2).^2)



%%%%%%%%%%%%%%%%%%


%% grid points
grid_option = 3;
%可以改
g=3000;
[grid,mnew] = select_grid(x_mean,grid_option,g);
%% L
[L,rowmax,removeind] = likelihood_matrix(x_mean,grid,x_cov_est,1);


options.scaleL = 0;
options.approxL = 0;
options.stoptol = 1e-6;
options.printyes = 0;
[~,weight,~,~,~,info,~] = DualALM(L,options);

%%%%%%%%%%%%%%%%%%


denominator = (L*weight);

numerator = zeros(n, 3);

numerator(:,1:2) = (L*(grid.*weight)) ;

numerator(:,3) = L * (grid(:,2).^2 .* weight);
    
pm = numerator ./ denominator;

X = [ones(length(pm),1) pm];

b_est = regress(y_minus_z, X);
