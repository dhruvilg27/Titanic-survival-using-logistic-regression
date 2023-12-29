clc;
clear;
close all;

Titanic_table = readtable('train.csv');
Titanic_data = (table2cell(Titanic_table));

class = cell2mat(Titanic_data(:,3));
survived = cell2mat(Titanic_data(:,2));
sex = Titanic_data(:,5);
age = cell2mat(Titanic_data(:,6));

% Replace NaN values in age by mean age
mean_age = mean(age,'omitnan');
age(isnan(age))= mean_age;

% Discretize age to bins
Age_Categorized = discretize(age, [0 10 25 50 100],'categorical',{'Under 10', '10-25', '25-50', 'Above 50'});


Age_Categorized = grp2idx(Age_Categorized);
sex = grp2idx(sex);
tbl = table(class, sex, Age_Categorized, survived);

%log_model = fitglm(tbl, 'Distribution','binomial');
%ypred = predict(log_model,tbl(:,1:end-1));
% to round the probabilities to 0 and 1 i.e.0:not survived, 1:survived
%ypred = round(ypred);
Confusion_Matrix = confusionmat(survived,ypred);
Accuracy = trace(Confusion_Matrix)/sum(Confusion_Matrix, 'all')