function [w, iterations] = perceptron_learn(data_in)
% perceptron_learn: Run PLA on the input data
% Inputs:  data_in is a matrix with each row representing an (x,y) pair;
%                 the x vector is augmented with a leading 1,
%                 the label, y, is in the last column
% Outputs: w is the learned weight vector; 
%            it should linearly separate the data if it is linearly separable
%          iterations is the number of iterations the algorithm ran for

D = size(data_in, 2) - 2;

x = data_in(:,1:D+1);
y = data_in(:,D+2);

w = zeros(1,D+1);
iterations = 0;

while(1)
    iterations = iterations + 1;
    mis = find(sign(x * w') .* y <= 0);
    if (numel(mis) == 0)
        break
    end
    ind = mis(randi(numel(mis)));
    w = w + x(ind,:) * y(ind);
end

end