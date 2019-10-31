N = 10000000;

X = 2*rand(N,2)-1;

mb = cell2mat(cellfun(@learn, num2cell(X,2), 'UniformOutput', false));

gbar = mean(mb,1);

eout_total = mean(eout(mb));

var_total = mean(vari(mb, gbar));

bias_total = eout(gbar);

x = -1:.01:1;
f = x.*x;
gbar_f = gbar(1)*x+gbar(2);

plot(x,f);
hold on;
plot(x,gbar_f);
legend('f(x)','$\bar{g}(x)$','Interpreter','latex');
hold off;
