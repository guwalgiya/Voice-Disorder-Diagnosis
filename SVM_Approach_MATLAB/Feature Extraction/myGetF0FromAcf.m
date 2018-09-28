function [f0] = myGetF0FromAcf(acfVector, fs)

%% Computes the pitch for a block of audio from the ACF vector
% Input:
%   acfVector:  (blockSize x 1) float vector, ACF of a block
%   fs:         float, sampling rate in Hz
% Output:
%   f0:         float, fundamental frequency of the block in Hz  

% check input dimensions
[~, n] = size(acfVector);
if (n>1)
    error('illegal input acfVector');
end
[m,n] = size(fs);
if (m ~= 1 && n ~=1)
    error('illegal fs');
end

%% Please insert your code here

% calculate derivative of acf
grad = diff(acfVector);

% calculate second derivative of acf
grad2 = diff(grad);

% make them the same length as acfVector
grad = [0; grad];
grad2 = [0; 0; grad2];

% detect local maxima based on 1st and 2nd derivatives
local_maxima = [];
for i = 2:length(grad)
    if grad(i) * grad(i-1) <= 0
        if grad2(i) < 0
            local_maxima = [local_maxima; i];
       end
    end 
end

% handle case if no local maxima are located 
if isempty(local_maxima)
    f0 = 0;
    return;
end

% find f0 based on the maximum of the local maximas 
[~, idx] = max(acfVector(local_maxima));
idx = local_maxima(idx);
f0 = fs / (idx - 1);

end
