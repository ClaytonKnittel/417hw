function [train_err, test_err] = AdaBoost(X_tr, y_tr, X_te, y_te, numTrees)
% AdaBoost: Implement AdaBoost with decision stumps as the weak learners. 
%           (hint: read the "Name-Value Pair Arguments" part of the "fitctree" documentation)
%   Inputs:
%           X_tr: Training data
%           y_tr: Training labels
%           X_te: Testing data
%           y_te: Testing labels
%           numTrees: The number of trees to use
%  Outputs: 
%           train_err: Classification error of the learned ensemble on the training data
%           test_err: Classification error of the learned ensemble on test data
% 
% You may use "fitctree" but not any inbuilt boosting function

y_tr = (y_tr == 3) - (y_tr ~= 3);
y_te = (y_te == 3) - (y_te ~= 3);

n = size(X_tr,1);
w = ones(n,1)/n; % init input weights
train_err = zeros(numTrees,1);
g = zeros(size(X_te,1),1);
test_err_v = zeros(numTrees,1);

for t = 1: numTrees
    % Train weak learner
    t_w = fitctree(X_tr, y_tr, 'Weights', w, 'minparent', n, 'prune', 'off', 'mergeleaves', 'off');
    
    test = t_w.predict(X_tr) ~= y_tr;
    % Compute weighted training error
    train_err(t) = w' * test;
    
    % Compute importance of h_t
    a_t = 0.5 * log((1 - train_err(t))/train_err(t));
    
    z_t = 2 * sqrt(train_err(t) * (1 - train_err(t)));
    % update weights
    w = (w / z_t) .* exp((2 * test - 1) * a_t);
    
%     for i = 1: size(X_te)
%         g(i) = g(i) + a_t * t_w.predict(X_te(i,:));
%     end
    g = g + a_t * t_w.predict(X_te);
    
    test_err_v(t) = sum(sign(g) ~= y_te, 1)/size(X_te,1);
end

test_err = test_err_v(numTrees);

% Plot out-of-bag-error 
plot(1:numTrees, test_err_v);
ylabel('Error');
xlabel('Num Trees (weak learners)');

hold on
plot(1:numTrees, train_err);
legend("Test Error", "Training Error");
hold off

train_err = train_err(size(train_err,1));

end