function v = vari(mb, gbar)

m = mb(:,1);
b = mb(:,2);

m0 = gbar(1);
b0 = gbar(2);

v = 1/3 * (m - m0).^2 + (b - b0).^2;

end

