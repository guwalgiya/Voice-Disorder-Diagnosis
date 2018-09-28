function [normFeatureMatrix,miu_vector,sd_vector] = Normalization(featureMatrix)

%% Performs z-score normalization over the input featureMatrix
%
% Inputs:
%   featureMatrix:  f x N float matrix, where f is the number of features (10 in this case)
%                   and N is the number of audio files in the directory.
%
% Outputs:
%   normFeatureMatrix:  f x N float matrix, where f is the number of features (10 in this case)
%                   and N is the number of audio files in the directory.

% Write your code below

[featureNum, songNum] = size(featureMatrix);
normFeatureMatrix = zeros(featureNum, songNum);
miu_vector = zeros(1,featureNum);
sd_vector = zeros(1,featureNum);
for i = 1 : featureNum
    miu = mean(featureMatrix(i,:));
    miu_vector(i) = miu;
    sd = std(featureMatrix(i,:));
    sd_vector(i) = sd;
    normFeatureMatrix(i,:) = (featureMatrix(i,:) - miu) / sd;
end
end
