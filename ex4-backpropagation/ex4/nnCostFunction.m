function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% 25 x 401
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% 10 x 26
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Number of examples
m = size(X, 1); % 5000

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Implement feedforward to get prediction for each of 5000 examples
A1 = [ones(m, 1) X]; % 5000 x 401
Z2 = A1 * Theta1'; % 5000 x 25
A2 = [ones(m, 1) sigmoid(Z2)];  % 5000 x 26
H = sigmoid(A2 * Theta2'); % 5000 x 10

% Expand the 'y' output (correct) values into a matrix of single values,
% every line of wich has 1 only at position corresponding to the
% required value of the example
% Another way to do: Y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
Y = eye(num_labels)(y,:);

% Compute difference between real values in Y and what we get from
% forward propagation in H
dif = -Y .* log(H) - (1-Y) .* log(1-H);

% For every example out of 5000, find cost
% First, sum of i=1..m rows
% Second, sum of k=1..K numbers
TH1 = Theta1(:, 2:end);
TH2 = Theta2(:, 2:end);
J = (1/m) * sum(sum(dif)) + (lambda/(2*m)) * (sum(sum(TH1 .^2)) + sum(sum(TH2 .^2)));

% -------------------------------------------------------------

% Accumulated values of backpropagation
D1 = zeros(size(Theta1_grad)); % 26x401 matrix
D2 = zeros(size(Theta2_grad)); % 10x26 matrix

% Backpropagation
delta3 = H - Y; % 5000 x 10
delta2 = delta3*Theta2(:,2:end) .* sigmoidGradient(Z2); % 5000 x 25

D1 = delta2'*A1; % 25 x 401
D2 = delta3'*A2; % 10 x 25

% Loop version of backpropagation (buggy)
% for t = 1:m
%
%   % 1. Take first example out of 5000 and
%   %    forward propagate this example
%   a1 = X(t,:); % 1x401 row
%
%   z2 = [1 a1 * Theta1']; % 1x26 row
%   a2 = sigmoid(z2);
%
%   z3 = a2 * Theta2'; % 1x10 row
%   a3 = sigmoid(z3);
%
%   % 2. Calculate delta3 which is a difference between
%   %    true value of Y and the
%   delta3 = a3 - Y(t,:); % 1x10 row
%
%   % 3. Same for hidden layer 2 but with another formula
%   %    and remove first column from delta2
%   delta2 = (delta3 * Theta2) .* sigmoidGradient(z2); % 1x26 row
%   delta2 = delta2(2:end); % 1x25 row
%
%   % 4. Accumulate the gradient for layers 1 and 2
%   D1 = D1 + delta2' * a1;
%   D2 = D2 + delta3' * a2;
% end

Theta1_grad = D1 / m;
Theta2_grad = D2 / m;

%Theta1_grad = Delta1 / m %+ lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m;
%Theta2_grad = Delta2 / m %+ lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
