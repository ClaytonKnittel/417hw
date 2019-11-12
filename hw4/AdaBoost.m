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
% train_err = zeros(numTrees,1);
g = zeros(size(X_te,1),1);
g_te = zeros(size(X_tr,1),1);
test_err_v = zeros(numTrees,1);
train_err_v = zeros(numTrees,1);

for t = 1: numTrees
    % Train weak learner
    t_w = fitctree(X_tr, y_tr, 'Weights', w, 'minparent', n, 'prune', 'off', 'mergeleaves', 'off');
    test = t_w.predict(X_tr) ~= y_tr;
    % Compute weighted training error
    epsilon = sum(w .* test);
    
%     train_err(t) = w' * test;
    
    % Compute importance of h_t
    a_t = 0.5 * log((1 - epsilon)/epsilon);
    z_t = 2 * sqrt(epsilon * (1 - epsilon));
    
    % update weights
    w = (w / z_t) .* exp(-a_t .* t_w.predict(X_tr) .* y_tr);
    
%     for i = 1: size(X_te)
%         g(i) = g(i) + a_t * t_w.predict(X_te(i,:));
%     end
    g = g + a_t * t_w.predict(X_te);
    g_te = g_te + a_t * t_w.predict(X_tr);
    train_err_v(t) = sum(sign(g_te) ~= y_tr, 1)/n;
    test_err_v(t) = sum(sign(g) ~= y_te, 1)/size(X_te,1);
end

test_err = test_err_v(numTrees);

train_err = train_err_v(numTrees);

% Plot out-of-bag-error 
plot(1:numTrees, test_err_v);
ylabel('Error');
xlabel('Num Trees (weak learners)');

hold on
plot(1:numTrees, train_err_v);
legend("Test Error", "Training Error");
hold off


end