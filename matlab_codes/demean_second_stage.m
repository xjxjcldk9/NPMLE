function [z_excluded, z_coef] = demean_second_stage(y, z)

% y: (n,m)
% z: (n, m, 2)

[n,m] = size(y);

y_mean = mean(y,2);
z_mean = mean(z,2);



y_demean = y - y_mean;
z_demean = z - z_mean;

y_demean_reshaped = reshape(y_demean, n*m,1);
z_demean_reshaped = reshape(z_demean, n*m,2);

z_coef = regress(y_demean_reshaped, z_demean_reshaped);

z_excluded = y_mean -  squeeze(z_mean)* z_coef;

end 