function line = learn(xs)

x1 = xs(1);
x2 = xs(2);

line = zeros(1,2);
line(1) = x1+x2;   % m
line(2) = -x1.*x2; % b

end

