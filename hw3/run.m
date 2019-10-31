clear variables;
data = csvread('cleveland_train.csv', 1, 0);

n = size(data, 2) - 1;
X = [ones(size(data, 1), 1) data(:, 1:n)];
y = 2*data(:, n+1) - 1;

Z = ones(size(data, 1), 1);
for x = 2:size(X,2)
    Z(:,x) = zscore(X(:,x));
end
X = Z;

n0 = 7.7;

t = cputime;
[t1, w1, e_in1] = logistic_reg(X, y, zeros(n+1,1), 10000, n0);
e1 = cputime-t;
t = cputime;
[t2, w2, e_in2] = logistic_reg(X, y, zeros(n+1,1), 100000, n0);
e2 = cputime - t;
t = cputime;
[t3, w3, e_in3] = logistic_reg(X, y, zeros(n+1,1), 1000000, n0);
e3 = cputime-t;

test = csvread('cleveland_test.csv', 1, 0);

nt = size(test, 2) - 1;
Xt = [ones(size(test, 1), 1) test(:, 1:nt)];
yt = test(:, nt+1);

Zt = ones(size(test, 1), 1);
for x = 2:size(Xt,2)
    Zt(:,x) = zscore(Xt(:,x));
end
Xt = Zt;

err1 = find_test_error(w1, Xt, yt);
err2 = find_test_error(w2, Xt, yt);
err3 = find_test_error(w3, Xt, yt);
