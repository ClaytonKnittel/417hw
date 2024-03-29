function [test_error] = find_test_error(w, X, y)

% find_test_error: compute the test error of a linear classifier w. The
%  hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
%  Inputs:
%		w: weight vector
%       X: data matrix (without an initial column of 1s)
%       y: data labels (plus or minus 1)
%
%  Outputs:
%        test_error: binary error of w on the data set (X, y) error;
%        this should be between 0 and 1.

cls = X * w > 0;

test_error = 1 - sum(cls == y) / size(y, 1);

end