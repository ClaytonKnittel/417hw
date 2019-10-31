function [num_iters, bounds_minus_ni] = perceptron_experiment(N, d, num_samples)
% perceptron_experiment: Code for running the perceptron experiment in HW1
% Inputs:  N is the number of training examples
%          d is the dimensionality of each example (before adding the 1)
%          num_samples is the number of times to repeat the experiment
% Outputs: num_iters is the # of iterations PLA takes for each sample
%          bound_minus_ni is the difference between the theoretical bound
%                         and the actual number of iterations
%          (both the outputs should be num_samples long)

bounds_minus_ni = zeros(num_samples);

wstar = [0 rand(1,d)];

tx = cat(2, ones(N,1,num_samples), 2*rand(N,d,num_samples)-1);
ty = sign(sum(tx.*wstar, 2));
tdata = reshape(num2cell(cat(2, tx, ty),[1 2]),1,[]);
[w, num_iters] = cellfun(@perceptron_learn,tdata,'UniformOutput',false);
num_iters = cell2mat(num_iters);

R = reshape(max(sqrt(sum(tx.*tx,2)),[],1),1,[]);
wabs = cellfun(@(x) x*x',w);

xs  = reshape(num2cell(tx,[1 2]),1,[]);
ys  = reshape(num2cell(ty,[1 2]),1,[]);
rho = cellfun(@(x,y,w) min(y.*(x*w')),xs,ys,w);

tmax = ceil((R.*wabs./rho).^2);
diff = tmax-num_iters;

hold on;
subplot(2,1,1);
histogram(log10(diff));
title('Logarithmic difference of number of iterations and the theoretical maximum');
xlabel('log_{10}(theoretical max - actual # iterations)');
ylabel('n');

subplot(2,1,2);
histogram(num_iters);%,.5:max(num_iters)+.5);
title(sprintf('Number of iterations to convergence of %d experiments', num_samples));
xlabel('Number of iterations');
ylabel('n');
hold off;

end