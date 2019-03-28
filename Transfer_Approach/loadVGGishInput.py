# ===============================================
# Import Packages and Functions
import numpy as np


# ===============================================
# Main Functions
def loadVGGishInput(selected_combo, classes, data, VGGish_input_shape, work_on_augmented, snippet_dict):
  
  
    # ===============================================
    # Initialization  
    label_1, label_2, label_3, snippet_num_list, distribution = [], [], [], [], {}
    

    # ===============================================
    # Format: distribution = {"Normal": [xxx, xxxxxxx], "Pathol" : [yyy, yyyyyy]} xxx, yyy are integers
    for a_class in classes:
        distribution[a_class] = [0,0]
    

    # ===============================================  
    for a_combo in selected_combo:
        

        # ===============================================
        # Basic Information, for a_combo in selected_combo:
        cur_name  = a_combo[0]
        cur_class = a_combo[1]


        # ===============================================   
        # label_1: 1, label_2:[0,1], label_3: pathol   
        num_snippet = snippet_dict[cur_name][0]
        cur_label_1 = snippet_dict[cur_name][1]
        cur_label_2 = snippet_dict[cur_name][2]
        cur_label_3 = snippet_dict[cur_name][3]
        

        # ===============================================   
        # Use that to track loaded data   
        snippet_num_list.append(num_snippet)


        # ===============================================   
        label_1 = label_1 + [cur_label_1] * num_snippet
        label_2 = label_2 + [cur_label_2] * num_snippet
        label_3 = label_3 + [cur_label_3] * num_snippet


        # ===============================================   
        distribution[cur_class][0] = distribution[cur_class][0] + 1
        distribution[cur_class][1] = distribution[cur_class][1] + num_snippet
    

    # ===============================================   
    # Prepare to load data
    loaded_data = np.zeros((sum(snippet_num_list), ) + VGGish_input_shape)
    start_index = 0
    end_index   = 0    
    for combo in selected_combo:


        # ===============================================   
        # Basic Information, for a_combo in selected_combo
        cur_name  = combo[0]
        cur_class = combo[1]
        end_index = end_index + snippet_dict[cur_name][0]
        

        # ===============================================  
        # Load augmented data or not
        if work_on_augmented:
            VGGish_inputs = [data_point[2] for data_point in data if (data_point[0] == cur_name)]
        else:
            VGGish_inputs = [data_point[2] for data_point in data if (data_point[0] == cur_name and data_point[1] == "N0.0")]
        
        
        # ===============================================
        for i in np.arange(start_index, end_index):
            loaded_data[i] = VGGish_inputs[i - start_index]


        # ===============================================  
        start_index = start_index + snippet_dict[cur_name][0]
    
    
    # ===============================================
    return [loaded_data, label_1, label_2, label_3, distribution, snippet_num_list]
