# ===============================================
# Import Packages and Functions
from   os     import walk
import pickle


# ===============================================
# MainFunction
def compressDictionary(dataset_path, classes, dsp_package, all_combo, slash, work_on_augmentated):
  

    # ===============================================
    snippet_length, snippet_hop, fft_length, fft_hop, mel_length = dsp_package
    

    # ===============================================
    snippet_amount_dict = {}


    # ===============================================
    # a_combo = [file_name, its_label]
    for a_combo in all_combo:


        # ===============================================
        # label_1: 1, label_2:[0,1], label_3: pathol
        original_file_name   = a_combo[0]
        class_index          = classes.index(a_combo[1])
        
        # ===============================================
        label_1              = class_index
        
        # ===============================================
        label_2              = [0] * len(classes)
        label_2[class_index] = 1                
        
        # ===============================================
        label_3              = a_combo[1]
        

        # ===============================================
        if work_on_augmentated:
            sub_folder = a_combo[1] + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        else:
            sub_folder = a_combo[1] + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented"

        
        # ===============================================
        wavfile_path = dataset_path + slash + sub_folder + slash + a_combo[0]


        # ===============================================
        wavfile_name_list = []
        for (dirpath, dirnames, filenames) in walk(wavfile_path):
            wavfile_name_list.extend(filenames)
            break
        

        # ===============================================
        snippet_amount = 0
        for wavfile_name in wavfile_name_list:
            snippet_amount = snippet_amount + 1


        # ===============================================
        snippet_amount_dict[original_file_name] = [snippet_amount, label_1, label_2, label_3]
        

        # =============================================== 
        print(original_file_name, label_1, snippet_amount)


    # ===============================================
    if work_on_augmentated:
        dict_file_name = "Dictionary_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"                 


    # ===============================================
    else:
        dict_file_name = "Dictionary_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented"
    

    # ===============================================
    # Same the dictionary as a pickle file
    with open(dataset_path + slash + dict_file_name + ".pickle", "wb") as temp_file:
        pickle.dump(snippet_amount_dict, temp_file, protocol = pickle.HIGHEST_PROTOCOL)

    
    # ===============================================
    print("Dictionary is saved")
    