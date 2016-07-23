function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of parameters

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Linear regression function (hypothesis)
H = X * theta; % 12x1

J = ( sumsq(H - y) + lambda * sumsq(theta(2:end)) ) / (2*m);

% =========================================================================

grad(1) =  (H - y)' * X(:, 1) / m;

for j = 2:n
  grad(j) = ((H - y)' * X(:, j) + lambda*theta(j))/m;
end

end
