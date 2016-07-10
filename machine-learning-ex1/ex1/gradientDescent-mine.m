function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)
    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    %   taking num_iters gradient steps with learning rate alpha

    % Initialize some useful values
    m = length(y);
    J_history = zeros(iterations, 1);

    function hiposis = h(Xi, theta)
        hiposis =  Xi * theta;
    end

    function s = sumDiff(X, theta, y, j)
        s = 0;
        for i = 1:length(y)
            s = s + (h(X(i,:), theta) - y(i,:)) * X(i, j);
        end
    end

    for iter = 1:iterations

        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %

        for j = 1:length(theta)
            theta(j) = theta(j) - (alpha / m) * sumDiff(X, theta, y, j);
        end

        %theta(1,1) = theta(1,1) - (alpha/m)*sum((X*theta-y));
        %theta(2,1) = theta(2,1) - (alpha/m)*sum((X*theta-y).*X(:,2));
        
        % ============================================================

        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);
    end

    plot(1:iterations, J_history);
end
