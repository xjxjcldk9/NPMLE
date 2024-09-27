function [b_est, grid, weight, theta, y, pm] = dim2_twostep(n, m, prior, x_cov, gamma, delta, beta)


[y, x, z, theta] = data_generator(n, m, prior, x_cov, gamma, delta, beta);


[x_mean_exclude_z, z_coef_likelihood, mean_err_cov] = demean_likelihood(x,z);


[y_mean_exclude_z, z_coef_second] = demean_second_stage(y,z);


%%%%%%%%%%%%%%%%%%


%% grid points
grid_option = 3;
%可以改
g=3000;
[grid,mnew] = select_grid(x_mean_exclude_z, grid_option, g);
%% L
[L,rowmax,removeind] = likelihood_matrix(x_mean_exclude_z, grid, mean_err_cov, 1);


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

b_est = regress(y_mean_exclude_z, X);
