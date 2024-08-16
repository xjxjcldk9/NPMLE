
function [b, grid, w] = matlab_wrapper(n)

m = 500;


theta = unifrnd(-3,3,n,1);



x = randn(n,1) + theta;




y = 1 + 2 * theta + randn(n,1);



ub = max(x) + eps;
lb = min(x) - eps;
grid = linspace(lb,ub,m);


diffM = x*ones(1,m) - ones(n,1)*grid;


L = normpdf(diffM);


options.maxiter = 1000;
options.stoptol = 1e-6;
options.stopop = 3;
options.printyes = 0;
options.approxL = 1;

[obj,w,~,u,v,info,runhist] = DualALM(L,options);




pm = (L*(grid'.*w)) ./ (L*w);



X = [ones(length(pm),1) pm];

b = regress(y, X);


