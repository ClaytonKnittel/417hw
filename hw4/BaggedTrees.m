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

n = size(X_tr, 1);

% oob_err_pts{1:n,1} = [2 3];
oob_err_pts = zeros(numBags,n);
test_err = 0;

for b=1:numBags
    samples = floor(n * rand(n, 1)) + 1;
    Db = X_tr(samples,:);
    yb = y_tr(samples,:);

    unused = setdiff(1:n, samples);

    tree = fitctree(Db,yb,'CrossVal','off');

    pred = tree.predict(X_tr(unused,:));
    oob_err_pts(b, unused) = pred;

%     err = (pred ~= real);
%     [pred real err]
%     oob_err_pts(unused) = oob_err_pts(unused) + (tree.predict(X_tr(unused,:)) ~= y_tr(unused,:));

    test_err = test_err + (tree.predict(X_te) ~= y_te);
end


oob_err = sum(oob_err_pts) / n;

test_err = test_err / n;


end
