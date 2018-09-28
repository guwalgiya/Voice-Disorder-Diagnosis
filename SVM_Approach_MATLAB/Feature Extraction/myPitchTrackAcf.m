function [f0, timeInSec] = myPitchTrackAcf(x, blockSize, hopSize, fs)

%% Blockwise Pitch Tracking based on ACF
% Input:
%   x:          (N x 1) float vector, input signal
%   blockSize:  int, block size of the blockwise process
%   hopSize:    int, hop size of the blockwise process
%   fs:         float, sample rate in Hz
% Output:
%   f0:         (numBlocks x 1) float vector, detected pitch (Hz) per block
%   timeInSec:  (numBlocks x 1) float vector, time stamp (sec) of each block

% Check input dimensions 
[~,n] = size(x);
if (n>1)
    error('illegal input signal');
end
[m,n] = size(blockSize);
if (m ~= 1 && n ~=1)
    error('illegal blockSize');
end
[m,n] = size(hopSize);
if (m ~= 1 && n ~=1)
    error('illegal hopSize');
end
[m,n] = size(fs);
if (m ~= 1 && n ~=1)
    error('illegal fs');
end

%% Please write your code here. Follow the steps as per the comments
% first block the audio and get the time-stamps per block (myBlockAudio)
[xb, timeInSec] = myBlockAudio(x, blockSize, hopSize, fs);

% assign memory for Output
f0 = zeros(size(xb,2), 1);

for i = 1:size(xb, 2)
	% compute the ACF for each block (myCompAcf)
	r = myCompAcf (xb(:, i), 1);
	
	% compute the pitch in Hz for each block (myGetF0FromACF)
	f0(i) = myGetF0FromAcf(r, fs);
end

end