function[y] = myLowPassFilter(x,fs,fc,order)
    
    [b,a] = butter(order, 2 * fc / fs,'low');
    %[h,w] = freqz(b,a);
    y = filter(b,a,x);
    
    %??
    % time = (1 : length(x)) /fs?
    time = (0 : length(x) - 1) / fs;
    
    subplot(2,1,1);
    scatter(time,x);
    subplot(2,1,2);
    scatter(time,y);
    
    
end