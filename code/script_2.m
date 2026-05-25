% Script 2: preparation of reference templates and test templates

clear;
clc;
close all;

% Loads allFeatures, allLabels and config.
load('../results/extracted_features.mat');

fprintf('Loaded: %d samples x %d features\n', size(allFeatures,1), size(allFeatures,2));

% Normalization
[normalizedFeatures, mu, sigma] = zscore(allFeatures);

% PART I: Reference and test templates
% Training: 80%
% Testing: 20%

referenceTemplate = cell(config.numUsers,1);   
testTemplate= cell(config.numUsers,1);  

rng(42);                                 
for user = 1:config.numUsers
    idx = find(allLabels==user);         
    n   = numel(idx);
    idx = idx(randperm(n));             
    

    splitPoint = round(0.80*n);         
    referenceIdx  = idx(1:splitPoint);
    testIdx = idx(splitPoint+1:end);
    
    referenceTemplate{user}  = normalizedFeatures(referenceIdx ,:);
    testTemplate{user} = normalizedFeatures(testIdx,:);
end

%% Create tensors to train the NN.
trainingRatio = 0.80;
validationRatio   = 0.10;
testRatio  = 0.10;

X_train = [];
Y_train = [];    % Holds class (user) label

X_validation   = []; 
Y_validation   = [];    % Holds class (user) label

X_test  = []; 
Y_test  = [];    % Holds class (user) label
L_test = [];  

for user = 1:config.numUsers
    idx = find(allLabels==user);
    n   = numel(idx);
    
    % No. of segments for training.
    a = floor(trainingRatio*n);

    % No. of segments for validation.
    b = floor(validationRatio*n);
    
    trainIdx = idx(1:a);
    valIdx   = idx(a+1:a+b);
    testIdx  = idx(a+b+1:end);
    
    % Appends feature vectors to training matrix.
    X_train = [X_train; normalizedFeatures(trainIdx,:)];
    % Appends user label for every segment.
    Y_train = [Y_train; repmat(user, size(trainIdx))];

    X_validation   = [X_validation;   normalizedFeatures(valIdx,:)];
    Y_validation   = [Y_validation;   repmat(user, size(valIdx))];

    X_test  = [X_test;  normalizedFeatures(testIdx,:)];
    Y_test  = [Y_test;  repmat(user, size(testIdx))];
    L_test  = [L_test;  allLabels(testIdx)];
end

% One-hot encoding
numClasses = config.numUsers;
ytrain = zeros(numel(Y_train), numClasses);
yval   = zeros(numel(Y_validation),   numClasses);
ytest  = zeros(numel(Y_test),  numClasses);

for k = 1:numClasses
    ytrain(:,k) = (Y_train==k);
    yval(:,k)   = (Y_validation==k);
    ytest(:,k)  = (Y_test==k);
end


save('../results/prepared_data.mat', ...
     'referenceTemplate', 'testTemplate', ...   
     'X_train','ytrain','X_validation','yval','X_test','ytest','L_test', ...
     'mu','sigma','config');

fprintf('Per-user templates saved (ref/test).\n');
fprintf('Global sets -> train: %d   val: %d   test: %d\n', ...
         size(X_train,1), size(X_validation,1), size(X_test,1));
