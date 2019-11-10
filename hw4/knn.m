

X = [2 0; 5 2; 6 3; 0 1; 2 3; 4 4];
y = [1; 1; 1; -1; -1; -1];
% D = {2, 0, '1' ; 5, 2, '1'; 6, 3, '1'; 0, 1, '-1'; 2, 3, '-1'; 4, 4, '-1'};

% KNN decision boundary for k = 1
knn1 = fitcknn(X, y, 'NumNeighbors' , 1);
xrange = 0:.01:6;
yrange = 0:.01:4;
[xg, yg] = meshgrid(xrange, yrange);
locs = [xg(:), yg(:)];

predictions = knn1.predict(locs);

subplot(2,1,1);
gscatter(xg(:), yg(:), predictions, 'rgb');
title('regular');


X(:,2) = X(:,2) * 5;
knn2 = fitcknn(X, y, 'NumNeighbors' , 1);
xrange = 0:.01:6;
yrange = 0:.05:20;
[xg, yg] = meshgrid(xrange, yrange);
locs = [xg(:), yg(:)];

predictions = knn2.predict(locs);

subplot(2,1,2);
gscatter(xg(:), yg(:), predictions, 'rgb');
title('scaled');