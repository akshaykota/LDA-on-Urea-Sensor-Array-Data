clear all
close all
clc

trainData = csvread('data2kfreq.csv'); % Load Training Data
testData = csvread('testdata2kfreq.csv'); % Load Testing Data

class = unique(trainData(:,end)); % Get the classes
Sw = zeros(size(trainData, 2)-1); % Initialization of within-class Scatter Matrix
Sb = zeros(size(trainData, 2)-1); % Initialization of between-class Scatter Matrix
dataMeanVec = mean(trainData(:, 1:end-1), 1); % Computing mean of each feature

% Looping over number of classes to update Sw and Sb
for i = 1:length(class)
    classBoolVec = trainData(:,end) == class(i); 
    classData = trainData(classBoolVec, 1:end-1); % Get the data of each class in a matrix
    classMeanVec(i, :) = mean(classData, 1); % Compute the mean of each feature of each class
    Sw = Sw + cov(classData); % Add the covariance matrix of each class to the within-class scatter matrix
    Sb = Sb + length(classData)*(classMeanVec(i, :) - dataMeanVec)'*...
        (classMeanVec(i, :) - dataMeanVec); % Add to between-class scatter matrix
end

[eigvec, eigval] = eig(inv(Sw)*Sb); % Computing the eigenvalues and eigenvectors of inv(Sw)*Sb

xin = testData(:,1:end-1)'; % Get the test data in matrix
transXin = eigvec(:,2)'*xin; % Transform the test data by multiplying with eigenvectors
transMeanVec = eigvec(:,2)'*classMeanVec'; % Transform mean of each class by multiplying with eigenvectors
disp('Predicted Classes');

% Looping over all test samples
for n = 1:size(transXin,2)
% Computing distances between transformed test sample and 
% transformed means of all the classes
dist1 = norm(transMeanVec(:,1)-transXin(:,n));
dist2 = norm(transMeanVec(:,2)-transXin(:,n));
dist3 = norm(transMeanVec(:,3)-transXin(:,n));
dist4 = norm(transMeanVec(:,3)-transXin(:,n));

% Finding the class mean which is nearest to the test sample
[minval, minidx] = min([dist1, dist2, dist3, dist4]);
outClass(n) = minidx;
disp(minidx);
end

% Computing the prediction accuracy
outClassBool = outClass' ~= testData(:,end);
numOfMisclassifications = length(testData(outClassBool));
numOfTestSamples = length(testData(:,end));
accuracy = (numOfTestSamples - numOfMisclassifications)/numOfTestSamples*100;

disp('Actual Classes');
disp(testData(:,end));
disp('Classification Accuracy');
disp(strcat(num2str(accuracy),'%'));

% Plotting of data
transXtrain = eigvec(:,2)'*trainData(:,1:2)';
figure, scatter(transXtrain', zeros(size(transXtrain,2),1), 25, trainData(:,end), 'filled');
title('Data after projection onto eigenvector corresponding to largest eigenvalue');
hold on
scatter(transMeanVec', zeros(size(transMeanVec,2),1), 100, [1 2 3 4], 'filled')

figure, scatter(trainData(:,1), trainData(:,2), 50, trainData(:,end), 'filled');
title('Scatter Plot of Initial Data');xlabel('Feature 1 (E.1) -->');
ylabel('Feature 2 (E.2) -->');