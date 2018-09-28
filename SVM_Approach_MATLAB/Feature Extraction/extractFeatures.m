function [featureVect] = extractFeatures(x, blockSize, hopSize, fs)

%% Blocks an audio signal and calls the individual feature extraction functions.
% 
% Input:
%   x:           N x 1 float vector, input signal
%   blockSize:   int, size of each block
%   hopSize:     int, hop size
%   fs:          int, sampling frequency of x
% Output:
%   featureVect: (5 x numBlocks) float matrix, where numBlocks is 
%                the number of blocks
% Note:
%   1)  numBlocks depends on the length of the audio, zeropadding may be needed

[~,n] = size(x);
if (n>1)
    error('Illegal input signal. Signal has to be downmixed');
end
[m,n] = size(blockSize);
if (m ~= 1 && n ~=1)
    error('Illegal blockSize');
end
[m,n] = size(hopSize);
if (m ~= 1 && n ~=1)
    error('Illegal hopSize');
end
[m,n] = size(fs);
if (m ~= 1 && n ~=1)
    error('Illegal fs');
end
%% Write your code below
% Features in Time Domain
[xb, ~] = myBlockAudio(x, blockSize, hopSize, fs);

%[vta,~] = FeatureTimeMaxAcf(x, blockSize, hopSize,fs);
[vtp,~] = FeatureTimePredictivityRatio(x,blockSize,hopSize,fs);
[vrms,~] = FeatureTimeRms(x,blockSize,hopSize,fs);
[vstd,~] = FeatureTimeStd(x,blockSize,hopSize,fs);
[vzc,~] = FeatureTimeZeroCrossingRate(x,blockSize,hopSize,fs);
 
% Features in Frequency Domain

[X, ~] = myComputeSpectrogram(xb, fs, blockSize);
[vmfcc] = FeatureSpectralMfccs(X,fs);
[vsc] = FeatureSpectralCentroid(X,fs);
[vtsc] = FeatureSpectralCrestFactor(X,fs);
[vsd] = FeatureSpectralDecrease(X,fs);
[vtf] = FeatureSpectralFlatness(X,fs);
[vsf] = FeatureSpectralFlux(X,fs);
[vsk] = FeatureSpectralKurtosis(X,fs);
vsk = vsk';
[vsr] = FeatureSpectralRolloff(X,fs);
[vssk] = FeatureSpectralSkewness(X,fs);
[vssl] = FeatureSpectralSlope(X,fs);
[vss] = FeatureSpectralSpread(X,fs);
[vtpr] = FeatureSpectralTonalPowerRatio(X,fs);

featureVect = [vtp;vrms;vstd;vzc;vsc;vtsc;vsd;vtf;vsf;vsk;vsr;vssk;vssl;vss;vtpr;vmfcc];
end