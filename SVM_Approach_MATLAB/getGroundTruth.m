function [main_names, main_classes] = getGroundTruth(data_main_path, classes) 
    original_amount = 0;
    for i = 1 :length(classes)
        data_sub_path  = char(strcat(data_main_path, '/', classes(i)));
        original_amount = length(dir(data_sub_path)) + original_amount - 2; % - 2 for .. and ...
    end

    main_names = strings(original_amount,1);
    main_classes = strings(original_amount,1);
    k = 1;
    for i = 1 : length(classes)
        data_sub_path  = char(strcat(data_main_path, '/', classes(i)));

        folder_content = dir(data_sub_path);
        for j = 3 : length(folder_content)
            a_name = folder_content(j).name;
            main_names(k) = a_name(1:length(a_name) - 4);
            main_classes(k) = classes(i);
            k = k + 1;
        end
    end
end