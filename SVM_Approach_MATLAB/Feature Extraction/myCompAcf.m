function [r] = myCompAcf (inputVector, bIsNormalized)

%% Computes the ACF of an input with optional normalization
% Input:
%   inputVector:	(N x 1) float vector, block of audio
%   bIsNormalized: 	bool variable for normalization of ACF 
% Output:
%   r:				(N x 1) float vector, ACF of the inputVector

% set uninitialized input arguments
if (nargin < 2)
    bIsNormalized = true;
end

% check input dimension
[m,n] = size(inputVector);
if ((m<=1 && n<=1) || (m>1 && n>1))
    error('illegal input vector');
end

%% Please insert your ACF computation code here

% Time-domain approach
%{
% allocate memory for result
r = zeros(size(inputVector));
for lag = 1:size(inputVector)
	for n = lag:size(inputVector)
    	r(lag) = r(lag) + inputVector(n)*inputVector(n-lag+1);
    end
end
%}
    
% Frequency-domain approach
zeropadInput = [inputVector; zeros(length(inputVector),1)]; % zeropad input vector to make it double the length
inputSpectrum = fft(zeropadInput); % compute fft
r = ifft(abs(inputSpectrum).^2); % compute ifft of squared magnitude spectrum
r = r(1:length(inputVector)); % truncate result to match the input length

% normalize result
if (bIsNormalized)
    %% Please inset your normalization code here
    normConst = sum(inputVector.^2); % calculate the normalization constant
    r = r / normConst; % normalize the output
end

end