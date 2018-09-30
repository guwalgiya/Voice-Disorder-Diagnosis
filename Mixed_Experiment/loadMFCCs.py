from   os      import walk
import numpy   as     np
import librosa
import math

def loadMFCCs(selected_combo, classes, dsp_package, dataset_path, data, augmented, snippet_dict):

    # =============================================================================
    fs, snippet_length, snippet_hop, fft_length, fft_hop, num_features = dsp_package
    label_1, label_2, label_3, snippet_num_list, distribution          = [], [], [], [], {}
    
    # =============================================================================
    #[0 originial files, 0 snippets]
    for a_class in classes:
        distribution[a_class] = [0,0]
    
    # =============================================================================   
    for combo in selected_combo:
        original_file_name  = combo[0]
        original_file_class = combo[1]

        # =============================================================================
        # label_1: 1, label_2:[0,1], label_3: pathol   
        num_snippet = snippet_dict[original_file_name][0]
        cur_label_1 = snippet_dict[original_file_name][1]
        cur_label_2 = snippet_dict[original_file_name][2]
        cur_label_3 = snippet_dict[original_file_name][3]
        
        # =============================================================================   
        snippet_num_list.append(num_snippet)
        label_1 = label_1 + [cur_label_1] * num_snippet
        label_2 = label_2 + [cur_label_2] * num_snippet
        label_3 = label_3 + [cur_label_3] * num_snippet

        # =============================================================================
        distribution[original_file_class][0] = distribution[original_file_class][0] + 1
        distribution[original_file_class][1] = distribution[original_file_class][1] + num_snippet
    
    # =============================================================================   
    loaded_data    = np.zeros((sum(snippet_num_list), num_features))

    # =============================================================================
    start_index    = 0
    end_index      = 0   
    for combo in selected_combo:
        original_file_name  = combo[0]
        original_file_class = combo[1]
        
        end_index           = end_index + snippet_dict[original_file_name][0]
        
        # =============================================================================
        if augmented:
            MFCCs = [data_point[2] for data_point in data if (data_point[0] == original_file_name)]
        else:
            MFCCs = [data_point[2] for data_point in data if (data_point[0] == original_file_name and data_point[1] == "N0.0")]

        # =============================================================================
        for i in np.arange(start_index, end_index):
            loaded_data[i] = MFCCs[i - start_index]

        start_index  = start_index + snippet_dict[original_file_name][0]     

    return [loaded_data, label_1, label_2, label_3, distribution, snippet_num_list]