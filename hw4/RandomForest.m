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

end
