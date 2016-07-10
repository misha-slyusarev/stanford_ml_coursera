function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% oldpager = PAGER('/dev/null');
% oldpso = page_screen_output(1);
% oldpoi = page_output_immediately(1);

% Add columns of ones to the matrix of input values
% This will indicate input values for bias nodes, and overall forms our
% a1 matrix of neural network values on the first lavel of the network
a1 = [ones(m,1) X];

% Get z2 which is a matrix of first layer values with applied theta 1
% parameters, then get a2 which is z2 activated by sigmoid function,
% finally add ones as bias nodes to the second layer
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

[mx, p] = max(a3, [], 2);

% =========================================================================

% PAGER(oldpager);
% page_screen_output(oldpso);
% page_output_immediately(oldpoi);

end
