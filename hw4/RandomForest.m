function [oob_err, test_err] = RandomForest(X_tr, y_tr, X_te, y_te, numBags, m)
% RandomForest: Learns an ensemble of numBags CART decision trees using a random subset of
%               the features at each split on the input dataset and also plots the 
%               out-of-bag error as a function of the number of bags
%       Inputs:
%               X_tr: Training data
%               y_tr: Training labels
%               X_te: Testing data
%               y_te: Testing labels
%               numBags: Number of trees to learn in the ensemble
% 				m: Number of randomly selected features to consider at each split
% 				   (hint: read the "Name-Value Pair Arguments" part of the fitctree documentation)
%      Outputs: 
%	            oob_err: Out-of-bag classification error of the final learned ensemble
%               test_err: Classification error of the final learned ensemble on test data
%
% You may use "fitctree" but not "TreeBagger" or any other inbuilt bagging function

n = size(X_tr,1);

oob_err_pts = zeros(n,1); % Holds aggregated predictions
oob_err_bags = zeros(numBags,1); % Holds OOB in terms of number of bags
test_err = 0;

for b=1:numBags
    % Create dataset by sampling n points from D with replacement
    samples = floor(n * rand(n, 1)) + 1;
    Db = X_tr(samples,:);
    yb = y_tr(samples,:);
    
    % Learn decision tree t_b
    t_b = fitctree(Db,yb,'CrossVal','off'); % Holdout??? with cvpartition???
    
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
    test_err = test_err + (t_b.predict(X_te) ~= y_te);
end

% Calculate final out of bag error
oob_err = oob_err_bags(numBags);

test_err = test_err / n;

% Plot out-of-bag-error 
plot(1:numBags, oob_err_bags);
ylabel('OOB Error');
xlabel('Num Bags');

end

