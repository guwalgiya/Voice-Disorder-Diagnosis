function [jitter, shimmer] = jitter_shimmer(x,fs,time_theta)
    
    x = smooth(x);
    theta = abs(min(x)) * 0.8;
    
    [peaks,locs] = findpeaks(-x, 'sortstr','descend');
    
    helper = [peaks, locs];
    helper = helper(helper(:,1) > theta,:);
 
    cycle_min_locs = helper(:,2);
    cycle_min_locs = sort(cycle_min_locs);
    for i = 1 : length(cycle_min_locs) - 1
        if (cycle_min_locs(i+1) - cycle_min_locs(i)) * 1 / fs <= time_theta
            cycle_min_locs(i) = -1;
        end
    end
    cycle_min_locs = cycle_min_locs(cycle_min_locs >0);
    
    
    
    
    cycle_max_locs = zeros(length(cycle_min_locs) - 1,1);
    peak_to_peak = zeros(length(cycle_min_locs) - 1,1);
    for i = 1 : length(cycle_min_locs) - 1
        [~, max_loc] = max(   x(cycle_min_locs(i):cycle_min_locs(i+1))  );
        cycle_max_locs(i) = max_loc + cycle_min_locs(i);
        peak_to_peak(i) = x(cycle_max_locs(i)) - x(cycle_min_locs(i + 1));
    end
      
    cycle_length_in_samples = diff(cycle_max_locs);
    cycle_length_in_time = cycle_length_in_samples * 1 / fs;  
    jitter = mean(abs(diff(cycle_length_in_time)));
    
    shimmer = zeros(length(peak_to_peak) - 1,1);
    for i = 1 : length(peak_to_peak) - 1
        shimmer(i) = log10(peak_to_peak(i+1) / peak_to_peak(i)) * 20;
    end
    shimmer = mean(shimmer);
end