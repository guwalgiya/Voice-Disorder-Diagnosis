function [xb, timeInSec] = myBlockAudio(x, blockSize, hopSize, fs)

%% Blocks the input audio signal into overlapping buffers
% Input:
%   x:          N*1 float vector, input signal
%   blockSize:  int, size of each block
%   hopSize:    int, hop size
%   fs:         float, sampling rate in Hz
% Output:
%   xb:         (blockSize x numBlocks) float matrix, where numBlocks is 
%               the number of blocks
%   timeInSec:  (numBlocks x 1) float vector, time stamp (sec) of each block   
% Note:
%   1)  numBlocks depends on the length of the audio, zeropadding may be needed

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

%% Please write your code here

% get signal length
signal_length = length(x);
    
% compute number of blocks in audio
num_blocks = ceil(signal_length/hopSize);
    
% zero pad incoming signal appropriately
zero_pad = num_blocks*hopSize + blockSize - signal_length;
x = [x; zeros(zero_pad, 1)];
    
% split signal into blocks
xb = zeros(blockSize, num_blocks); % initialize output matrix
for i = 1:num_blocks
    xb(:, i) = x((i-1)*hopSize+1:(i-1)*hopSize + blockSize);
end

% create time stamps 
timeInSec = 1:hopSize:(num_blocks - 1)*hopSize + 1;
timeInSec = timeInSec - 1;
timeInSec = timeInSec' / fs;


end