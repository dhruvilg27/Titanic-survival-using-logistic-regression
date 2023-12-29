function [J,grad] = cost_function(theta,X,y)

J=0;
grad= zeros(size(theta)); 
m = size(X,1);


z = X*theta;
H = exp(z)./(exp(z) + 1);

J = y.*(log(H));
J = J + (1-y).*(log(1-H));
J = sum(J);
J = J*(-1/m);

grad = X'*(H-y);
grad = grad*(1/m);