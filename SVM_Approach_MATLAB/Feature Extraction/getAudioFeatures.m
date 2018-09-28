function getAudioFeatures(num_features,data_main_path, working_path, main_names, main_classes, package)
snippet_length = package(1);
snippet_hop = package(2);
block_size = package(3);
hop_size = package(4);

for i = 1 : length(main_names)
    disp(i)
    sub_folder_name  = strcat(main_classes(i),'_', string(snippet_length), 'ms_', string(snippet_hop),'ms');
    sub_path    = char(strcat(data_main_path, '/', sub_folder_name));
    snippet_path = char(strcat(sub_path, '/', main_names(i)));
    save_folder_name = char(strcat(main_names(i), '_features_block',string(block_size),'_hop',string(hop_size)));
    cd(sub_path)
    mkdir(save_folder_name)
    cd(working_path)
    
    snippet_content = dir(snippet_path);
    for j = 3 : length(snippet_content)
        snippet = char(strcat(snippet_path, '/', snippet_content(j).name));
        [x,fs] = audioread(snippet);
        voiceFeatureMatrix = extractFeatures(x,block_size,hop_size,fs);
        aggFeatureVector = aggregateFeaturesPerFile(voiceFeatureMatrix, num_features); %this vector is horizontal
        
        vector_name = snippet_content(j).name;
        vector_name = char(strcat(vector_name(1: length(vector_name) - 4), '.mat'));
        
        save_path = char(strcat(sub_path,'/',save_folder_name));
        
        cd(save_path)
        save(vector_name,'aggFeatureVector');
        cd(working_path)
    end
    disp(main_names(i))
    disp('................................................................')
    
    
end
