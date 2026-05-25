% Script 1: data segmentation and feature extraction

clear all;
close all;
clc;

config.numUsers = 10;
config.windowSize = 150;
config.overlap = 75;
config.samplingRate = 30;  % Samples per second.
config.data_path = '../data/';  % Path to CSV files with gait data.

% Creates place to save output.
if ~exist('../results', 'dir')
    mkdir('../results');
end

% PART I: Loads files and reads data.
gaitData = cell(config.numUsers, 2);

for user = 1:config.numUsers
    filename_fd = sprintf('%sU%dNW_FD.csv', config.data_path, user);
    filename_md = sprintf('%sU%dNW_MD.csv', config.data_path, user);
    
    % Reads CSV files, creates an array and stores it.
    gaitData{user, 1} = readmatrix(filename_fd);
    gaitData{user, 2} = readmatrix(filename_md);
    
    % No. of samples per user per file.
    fprintf('  User %d: %d + %d samples\n', user, size(gaitData{user,1},1), size(gaitData{user,2},1));
end

% PART II: Segmentation and feature extraction
allFeatures = [];
allLabels = [];

for user = 1:config.numUsers
    for session = 1:2

        % Extracts x, y, and z-axes for acceleration.
        accelerationData = gaitData{user, session}(:, 2:4);

        % Extracts x, y, and z-axes for gyroscope.
        gyroscopeData = gaitData{user, session}(:, 5:7);

        features = extract_features_from_windows(accelerationData, gyroscopeData, config.windowSize, config.overlap);
        labels = ones(size(features, 1), 1) * user;
        
        allFeatures = [allFeatures; features];
        allLabels = [allLabels; labels];
    end
    
end

fprintf('\nTotal: %d samples x %d features\n', size(allFeatures,1), size(allFeatures,2));


% Save all extracted features in a .mat file.
save('../results/extracted_features.mat', 'allFeatures', 'allLabels', 'config', '-v7');
