function [b_est, grid, weight, theta, y, pm] = dim2_twostep(n, prior, second_stage, x_cov, b)


% Can set different theta generation

%弄成3個不同長相的。1. uniform, 2. normal, 3. 長相奇特的

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

    

 
x = theta + mvnrnd(zeros(1,2),x_cov,n);

y = b(1) + theta(:,1) * b(2) + randn(n,1);

switch  second_stage
    case 'square'
        second_term = theta(:,2).^2;
    case 'log'
        second_term =  log(theta(:,2)+100);
    case 'cross'
        second_term = theta(:,1) .* theta(:,2);
    otherwise
        second_term =  theta(:,2);
end

y = y + second_term * b(3);




%% grid points
grid_option = 3;
%可以改
m=3000;
[grid,mnew] = select_grid(x,grid_option,m);
%% L
[L,rowmax,removeind] = likelihood_matrix(x,grid,x_cov,1);


options.scaleL = 0;
options.approxL = 0;
options.stoptol = 1e-6;
options.printyes = 0;
[~,weight,~,~,~,info,~] = DualALM(L,options);



denominator = (L*weight);


numerator = (L*(grid.*weight)) ;
switch  second_stage
    case 'square'
        numerator(:,2) = L * (grid(:,2).^2 .* weight);
    case 'log'
        %不要加100
        numerator(:,2) = L * (log(grid(:,2)+100) .* weight);
    case 'cross'
        numerator(:,2) = L * (grid(:,1) .* grid(:,2) .* weight);
end


pm = numerator ./ denominator;

X = [ones(length(pm),1) pm];

b_est = regress(y, X);
