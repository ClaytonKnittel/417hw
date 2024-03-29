function [oob_err, test_err] = BaggedTrees(X_tr, y_tr, X_te, y_te, numBags)
% BaggedTrees: Learns an ensemble of numBags CART decision trees on the input dataset 
%              and also plots the out-of-bag error as a function of the number of bags
%      Inputs:
%              X_tr: Training data
%              y_tr: Training labels
%              X_te: Testing data
%              y_te: Testing labels
%              numBags: Number of trees to learn in the ensemble
%     Outputs: 
%	           oob_err: Out-of-bag classification error of the final learned ensemble
%              test_err: Classification error of the final learned ensemble on test data
%
% You may use "fitctree" but not "TreeBagger" or any other inbuilt bagging function

y_tr = (y_tr == 3) - (y_tr ~= 3);
y_te = (y_te == 3) - (y_te ~= 3);

n = size(X_tr,1);

oob_err_pts = zeros(n,1); % Holds aggregated predictions
oob_err_bags = zeros(numBags,1); % Holds OOB in terms of number of bags
test_vote = zeros(size(y_te,1),1);

for b=1:numBags
    % Create dataset by sampling n points from D with replacement
    samples = floor(n * rand(n, 1)) + 1;
    Db = X_tr(samples,:);
    yb = y_tr(samples,:);
    
    % Learn decision tree t_b
    t_b = fitctree(Db,yb,'CrossVal','off');
    
    % Validation -> Retrieve all training pnts x_i not used to train
    unused = setdiff(1:n, samples);
    curr_pred = t_b.predict(X_tr(unused,:));
    
    % Save and aggregate prediction results 
    oob_err_pts(unused) = oob_err_pts(unused) + curr_pred;
    
    % Compute plurality vote of the trees that do not train on that point 
    oob_err_vote = sign(oob_err_pts);
    
    % Find and deal with ties
    tie = ~oob_err_vote;
    tie_dimen = size(tie,2);
    % Breaking tie vote arbitrarily
    % oob_err_vote(tie) = -1; % 
    oob_err_vote(tie) = datasample([-1,1],tie_dimen);
    
    % save each tree's OOB error, which accumulates as we move on and 
    % include the next tree in voting etc...
    oob_err_bags(b) = sum(oob_err_vote ~= y_tr)/n;
    
    % Plurality vote vs actual labels aggre for test error calc
    test_vote = test_vote + t_b.predict(X_te);
end

% Calculate final out of bag error
oob_err = oob_err_bags(numBags);

test_vote = sign(test_vote);
tie = ~test_vote;
tie_dimen = size(tie,2);
% Breaking tie vote arbitrarily
% oob_err_vote(tie) = -1; % 
test_vote(tie) = datasample([-1,1],tie_dimen);
test_err = sum(test_vote ~= y_te,1) / size(y_te,1);

% Plot out-of-bag-error 
plot(1:numBags, oob_err_bags);
ylabel('OOB Error');
xlabel('Num Bags');

end
