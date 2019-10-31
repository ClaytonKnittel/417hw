function [t, w, e_in] = logistic_reg(X, y, w_init, max_its, eta)

% logistic_reg: learn a logistic regression model using gradient descent
%  Inputs:
%       X:       data matrix (without an initial column of 1s)
%       y:       data labels (plus or minus 1)
%       w_init:  initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta:     learning rate
%
%  Outputs:
%        t:    the number of iterations gradient descent ran for
%        w:    learned weight vector
%        e_in: in-sample (cross-entropy) error

N = size(X, 1);

min_grad = .000001;

w = w_init;

for t = 1:max_its
    g = 1/N * sum((y.*X)' ./ (1 + exp(y .* (X * w)))', 2);
    if max(abs(g)) < min_grad
        break;
    end
    w = w + eta * g;
end


    function a = theta(s)
        a = 1 ./ (1 + exp(-s));
    end

e_in = sum(-(y == 1) .* log(theta(X * w)) - (y == -1) .* log(1 - theta(X * w)), 1);

end