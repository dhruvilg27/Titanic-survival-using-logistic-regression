function p = predict(X,theta)

p = (exp(X*theta)) ./ (exp(X*theta) + 1);


m = size(X,1);

for i=1:m
    if p(i)>=0.5
        p(i)=1;
    else p(i)=0;
    end
end