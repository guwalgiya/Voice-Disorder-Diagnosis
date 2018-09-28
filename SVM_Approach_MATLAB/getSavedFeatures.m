function [aug_data, aug_label] = getSavedFeatures(main_names, main_label, package, num_features, data_main_path)
snippet_length = package(1);
snippet_hop    = package(2);
block_size     = package(3);
hop_size       = package(4);

count = 0;
for i = 1 : length(main_names)
    sub_folder     = strcat(main_label(i) + '_' + string(snippet_length) + 'ms_' + string(snippet_hop) + 'ms');
    snippet_folder = strcat(main_names(i) + '_features_block' + string(block_size) + '_hop' + string(hop_size));
    
    snippet_path   = char(strcat(data_main_path, '/', sub_folder, '/', snippet_folder));
    count          = count + length(dir(snippet_path)) - 2;
end

aug_data  = zeros(count, num_features * 2);
aug_label = strings(count, 1);

k = 1;
for i = 1 : length(main_names)
    sub_folder     = strcat(main_label(i) + '_' + string(snippet_length) + 'ms_' + string(snippet_hop) + 'ms');
    snippet_folder = strcat(main_names(i) + '_features_block' + string(block_size) + '_hop' + string(hop_size));
    
    snippet_path   = char(strcat(data_main_path, '/', sub_folder, '/', snippet_folder));
    
    snippets = dir(snippet_path);
    for j = 3 : length(snippets)
        x = load(char(strcat(snippet_path, '/', snippets(j).name)));
        aug_data(k,:) = x.aggFeatureVector;
        aug_label(k)  = main_label(i);
        k = k + 1;
    end
     
end
end