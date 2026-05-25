function features = extract_features_from_windows(accelerationData, gyroscopeData, windowSize, overlap)

    % windowSize: number of samples in each window.
    % overlap: no. of overlapping samples.
    % stepSize: how far we will move the window forward.
    % This creates an overlap of 50%.
    stepSize = windowSize - overlap;
    
    % No. of samples in the data.
    numSamples = size(accelerationData, 1);
    % No. of segments within a signal.
    numWindows = floor((numSamples - windowSize) / stepSize) + 1;
    
    % We extract 15 features for acceleration (11 TD, 2 FD, 2 signal
    % shape), another 15 for gyroscope, and 3 magnitude features.
    numFeatures = 93;
    % features: an array where each row has a segment and each column has a feature.
    features = zeros(numWindows, numFeatures);
    
    for win = 1:numWindows
        start_idx = (win - 1) * stepSize + 1;
        end_idx = start_idx + windowSize - 1;
        
        % Extracts data for the segment.
        accelerationWindow = accelerationData(start_idx:end_idx, :);
        gyroscopeWindow = gyroscopeData(start_idx:end_idx, :);
        
        feat_idx = 1;
        
        % Extracting features from acceleration data.
        for axis = 1:3
            signal = accelerationWindow(:, axis);
            
            features(win, feat_idx) = mean(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = std(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = var(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = min(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = max(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = max(signal) - min(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = rms(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = median(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = iqr(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = skewness(signal);
            feat_idx = feat_idx + 1;
            
            % Kurotsis: sharpness of a peak.
            features(win, feat_idx) = kurtosis(signal);
            feat_idx = feat_idx + 1;
            
            zero_crossings = sum(abs(diff(sign(signal)))) / 2;
            features(win, feat_idx) = zero_crossings / length(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = mad(signal, 1);

            feat_idx = feat_idx + 1;
            
            % Applies Fast Fourier Transform.
            fft_signal = fft(signal);

            % Energy.
            fft_magnitude = abs(fft_signal);
            
            % Keeps only positive frequencies.
            half_length = floor(length(fft_magnitude) / 2);
            fft_magnitude = fft_magnitude(1:half_length);
            
            [~, peak_idx] = max(fft_magnitude);
            features(win, feat_idx) = peak_idx;
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = sum(fft_magnitude .^ 2);
            feat_idx = feat_idx + 1;
        end
        
        
        % Extracting features from gyroscope data.
        for axis = 1:3
            signal = gyroscopeWindow(:, axis);
            
            features(win, feat_idx) = mean(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = std(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = var(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = min(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = max(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = max(signal) - min(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = rms(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = median(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = iqr(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = skewness(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = kurtosis(signal);
            feat_idx = feat_idx + 1;
            
            zero_crossings = sum(abs(diff(sign(signal)))) / 2;
            features(win, feat_idx) = zero_crossings / length(signal);
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = mad(signal, 1);

            feat_idx = feat_idx + 1;
            
            % Applies Fast Fourier Transform.
            fft_signal = fft(signal);

            % Energy.
            fft_magnitude = abs(fft_signal);
            
            % Keeps only positive frequencies.
            half_length = floor(length(fft_magnitude) / 2);
            fft_magnitude = fft_magnitude(1:half_length);
            
            [~, peak_idx] = max(fft_magnitude);
            features(win, feat_idx) = peak_idx;
            feat_idx = feat_idx + 1;
            
            features(win, feat_idx) = sum(fft_magnitude .^ 2);
            feat_idx = feat_idx + 1;
        end
        

        % Calculate the overall strength of the movement.
        magnitude = sqrt(sum(accelerationWindow .^ 2, 2));
        features(win, feat_idx) = mean(magnitude);
        feat_idx = feat_idx + 1;
        
        features(win, feat_idx) = std(magnitude);
        feat_idx = feat_idx + 1;
        
        sma = sum(sum(abs(accelerationWindow))) / windowSize;
        features(win, feat_idx) = sma;
    end
    
    features(isnan(features)) = 0;
    
    features(isinf(features)) = 1e10;
    
end