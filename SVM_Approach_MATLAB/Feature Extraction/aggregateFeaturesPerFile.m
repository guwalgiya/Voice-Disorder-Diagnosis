function [aggregatedFeatureVec] = aggregateFeaturesPerFile(featureValues,numFeature)

%% Aggregates the values in featureValues and returns the mean and standard
% deviation, hence given 2 aggregated features per blocked feature per file.
% Input:
%   featureValues: 5 x numBlocks float array, features returned from extractFeatures.
% Output:
%   aggregatedFeatureVec: 10 x 1 float vector, aggregated feature vector
%                  Contains the mean and std of each in feature featureValues.

% Write your code below:
aggregatedFeatureVec = [];
for i = 1 : numFeature
    feature = featureValues(i,:);
    aggregatedFeatureVec = [aggregatedFeatureVec,mean(feature),std(feature)];
end