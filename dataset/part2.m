clear ; close all; clc              %clean everything

x1 = xlsread('train.csv','C1:C714'); %imported "pclass" as x1"
x2 = xlsread('train.csv','n1:n714'); % "age as x2"
x3 = xlsread('train.csv','G1:G714'); % Sibling or Spouse
x4 = xlsread('train.csv','h1:h714'); % parent children
x5 = xlsread('train.csv','j1:j714'); % Fare
x6 = xlsread('train.csv','m1:m714');  % male=0, female=1

m = length(x1);

X = [ ones(m,1) x1 x2 x3 x4 x5 x6 ];  % combine all features into one matrix

%mapping


y = xlsread('train.csv','b1:b714');   % output of training set

n = size(X,2); % no. of features including bias term

%theta = zeros(n,1); % initialize theta

%cost function



initial_theta = zeros(n,1);

[cost, grad] = cost_function(initial_theta, X, y);    % call the costfunction
fprintf('Cost at initial theta=%f \n', cost);
fprintf(' %f \n', grad);

options = optimset('Gradobj','on', 'MaxIter', 400);

[theta,cost] = fminunc(@(t)(cost_function(t,X,y)), initial_theta, options);

fprintf('\n Optimized cost =%f\n', cost);

%training accuracy 

p = predict(X,theta);     % call the predict function

fprintf('Train Accuracy: %f\n', (sum(p == y)/m) * 100);

%testing

xt1 = xlsread('test.csv','b2:b419'); %imported "pclass" as x1"
xt2 = xlsread('test.csv','L2:L419'); % "age as x2"
xt3 = xlsread('test.csv','f2:f419'); % Sibling or Spouse
xt4 = xlsread('test.csv','g2:g419'); % parent children
xt5 = xlsread('test.csv','i2:i419'); % Fare
xt6 = xlsread('test.csv','M2:M419');  % male=0, female=1

m_test = size(xt1,1);
X_test = [ ones(m_test,1) xt1 xt2 xt3 xt4 xt5 xt6 ];

p_test = predict(X_test,theta);     % get the results of the test data

fprintf('The survival prediction is: \n');  
disp(p_test);