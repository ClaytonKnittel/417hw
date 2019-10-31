function e = eout(mb)

m = mb(:,1);
b = mb(:,2);

e = 1/3 * m.^2 - 2/3 * b + b.^2 + 1/5;

end

