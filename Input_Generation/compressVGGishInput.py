# ===============================================
# Import Packages and Functions
from   os     import walk
import numpy  as     np
import pickle


# ===============================================
# Main function of this file
def compressVGGishInput(dataset_path, classes, dsp_package, all_combo, slash):
    

    # ===============================================
    snippet_length, snippet_hop, _, _, _ = dsp_package
    

    # ===============================================
    data = []


    # ===============================================
    # a_combo = [file_name, its_label], example: "[aaa, "Pathol"]"
    for a_combo in all_combo:


        # ===============================================
        cur_name = a_combo[0]


        # ===============================================
        sub_folder         = a_combo[1]   + "_"             + str(snippet_length) + "ms_"  + str(snippet_hop) + "ms"
        VGGishInput_folder = a_combo[0]   + "_VGGish_Input"
        VGGishInput_path   = dataset_path + slash           + sub_folder          + slash  + VGGishInput_folder


        # ===============================================
        VGGishInput_name_list = []
        for (dirpath, dirnames, filenames) in walk(VGGishInput_path):
            VGGishInput_name_list.extend(filenames)
            break
        

        # ===============================================
        snippet_amount = 0
        for VGGishInput_name in VGGishInput_name_list:


            # ===============================================
            VGGishInput = np.load(VGGishInput_path + slash + VGGishInput_name)


            # ===============================================
            snippet_amount = snippet_amount + 1


            # ===============================================
            # [file_name, pitch shift by semitone, mel_spectrogram] example: ["aaa", "N0.0", VGGishInput]
            data.append([cur_name, VGGishInput_name.split('_')[1], VGGishInput])
        
        
    # ===============================================
    print(len(data))
    

    # ===============================================
    data_file_name = "VGGish_Input_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
    

    # ===============================================
    temp_file = open(dataset_path + slash + data_file_name + '.pickle', 'wb')
    
    
    # ===============================================
    pickle.dump(data, temp_file)

    
    # ===============================================
    print("VGGisn Inputs are saved as a Pickle File")