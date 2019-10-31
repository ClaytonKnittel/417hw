clear N mu;

N = 6;
mu = 0.5;

X=0:.002:1;
Y=4*exp(-2*N*(X.*X));
A=zeros(1,size(X,2));

for i=1:size(X,2)
    ep = X(i);
    if ep<1/6
        A(i) = 231/256;
    elseif ep<1/3
        A(i) = 399/1024;
    elseif ep<1/2
        A(i) = 63/1024;
    else
        A(i) = 0;
    end
end

plot(X,A);
hold on;
xlabel('\epsilon');
ylabel('P[|\mu-\nu|>\epsilon]');

plot(X,Y);
legend('A(\epsilon)','Hoeffding bound');
hold off;

