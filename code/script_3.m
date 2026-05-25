% Script 3: rain classifier, test classifier, calculate accuracy (FAR, FRR, EER), implement NN
clear; 
close all; 
clc;

% Load data prepared in Script 2
load('../results/prepared_data.mat');   

if ~exist('../figures','dir')
    mkdir('../figures'); 
end


% Run PCA on the full training set
[coeff, ~, ~, ~, explained] = pca(X_train);
% Calculate cumulative variance
cumulativeVariance = cumsum(explained);
% Find number of components for 95% variance
numComponents = find(cumulativeVariance >= 95, 1, 'first');
% Keep only those components
coeff = coeff(:, 1:numComponents);

X_train_pca = X_train * coeff;
X_validation_pca = X_validation * coeff; 
X_test_pca  = X_test * coeff;

fprintf('PCA keeps %d components (%.1f %% variance)\n\n', numComponents, cumulativeVariance(numComponents));

% PART I: Building and training Binary Classifiers
numUsers = config.numUsers;
SVMModels = cell(numUsers,1);         

fprintf('Training SVMs for %d users...\n', numUsers);
for u = 1:numUsers
    % Genuine user: 1, Impostor: 0
    yb = double(ytrain(:,u)==1);      
    
    % Train SVM
    % We use a Radial Basis Function (RBF) kernel which is similar to a Neural Net
    % 'ClassNames' ensures column 2 is always the Positive class
    SVMModels{u} = fitcsvm(X_train_pca, yb, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'ClassNames', [0, 1]);
        
    % Calculate posterior probabilities (optional, helps with scoring)
    SVMModels{u} = fitPosterior(SVMModels{u}); 
    
    fprintf('.');
end
fprintf('\nTraining complete.\n');

% PART II: Obtain scores per user
scoreTest = zeros(size(X_test_pca,1), numUsers);   
for u = 1:numUsers
    % Predict scores on test data
    % [label, score] = predict(...)
    % score(:,2) is the posterior probability of being the Genuine User (Class 1)
    [~, scores] = predict(SVMModels{u}, X_test_pca);
    scoreTest(:,u) = scores(:,2); 
end

% PART III: Calculate metrics
FAR_all = zeros(numUsers,1);
FRR_all = zeros(numUsers,1);
EER_all = zeros(numUsers,1);
thresh_all = zeros(numUsers,1);

figure('Position',[100 100 1200 600]); 

for u = 1:numUsers
    % Compare scores against true labels (L_test)
    genuineScore    = scoreTest(L_test==u, u);  
    impostorScore   = scoreTest(L_test~=u, u);  
    
    thr  = linspace(0, 1, 1000);
    FAR  = arrayfun(@(t) mean(impostorScore >= t), thr);
    FRR  = arrayfun(@(t) mean(genuineScore < t), thr);
    
    % Find EER (where FAR is closest to FRR)
    [~,idxEER] = min(abs(FAR - FRR));
    FAR_all(u) = FAR(idxEER);
    FRR_all(u) = FRR(idxEER);
    EER_all(u) = (FAR(idxEER) + FRR(idxEER)) / 2;
    thresh_all(u) = thr(idxEER);
    
    % Plot ROC curves
    subplot(2, 5, u);
    plot(FAR*100, (1-FRR)*100, 'b-', 'LineWidth', 1.5); hold on;
    plot(FAR_all(u)*100, (1-FRR_all(u))*100, 'ro', 'MarkerFaceColor', 'r');
    grid on; 
    xlabel('FAR (%)'); ylabel('TAR (%)');
    title(sprintf('User %d (EER=%.2f%%)', u, EER_all(u)*100));
end

sgtitle('ROC Curves (PCA + SVM)');
saveas(gcf,'../figures/roc_curves.png');

% Summarise results 
avg_FAR = mean(FAR_all)*100;  
avg_FRR = mean(FRR_all)*100;  
avg_EER = mean(EER_all)*100;

fprintf('\nAuthentication Metrics (User-specific EER)\n');
fprintf('User |  FAR   |  FRR   |  EER\n');
fprintf('-----|--------|--------|--------\n');
for u = 1:numUsers
    fprintf(' %2d  | %5.2f%% | %5.2f%% | %5.2f%%\n', u, FAR_all(u)*100, FRR_all(u)*100, EER_all(u)*100);
end
fprintf('-----|--------|--------|--------\n');
fprintf('AVG  | %5.2f%% | %5.2f%% | %5.2f%%\n\n', avg_FAR, avg_FRR, avg_EER);

% Save results
results.FAR = FAR_all;
results.FRR = FRR_all;
results.EER = EER_all;
results.avg_FAR = avg_FAR;
results.avg_FRR = avg_FRR;
results.avg_EER = avg_EER;
results.pca_var_kept = sum(explained);
results.svm_models = SVMModels; % Saved SVM models instead of 'net'
results.coeff = coeff;
results.mu = mu;
results.sigma = sigma;

save('../results/results.mat', 'results');

T = table((1:numUsers)', FAR_all*100, FRR_all*100, EER_all*100, ...
          'VariableNames', {'User','FAR_percent','FRR_percent','EER_percent'});
writetable(T, '../results/auth_metrics.csv');

