function [z_excluded, z_coef] = demean_est(y, z)



y_mean = mean(y,2);
z_mean = mean(z,2);


y_demean = y - y_mean;
z_demean = z - z_mean;



z_coef = sum(y_demean .* z_demean) / sum(z_demean.^2);



z_excluded = y_mean - z_coef * z_mean;


end 