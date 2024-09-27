function [z_excluded, z_coef, mean_err_cov] = demean_likelihood(y, z)


%y: (n,m,2)
%z: (n,m,2)
[n,m,~] = size(y);

y_mean = mean(y,2);
z_mean = mean(z,2);


y_demean = y - y_mean;
z_demean = z - z_mean;


z_coef = sum(y_demean(:) .* z_demean(:)) / sum(z_demean(:).^2);


mean_err_cov = cov(reshape(y_demean - z_coef * z_demean, n*m, 2)) / double(m-1) ;

z_excluded = squeeze(y_mean - z_coef * z_mean);

end 