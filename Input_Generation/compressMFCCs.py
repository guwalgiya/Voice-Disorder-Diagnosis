from   os      import walk
import numpy   as     np
import librosa
import pickle

# =============================================================================
# Main function of this file
def compressMFCCs(dataset_path, classes, dsp_package, combo):
    
    # =============================================================================
    snippet_length, snippet_hop, fft_length, fft_hop, _ = dsp_package
    
    # =============================================================================
    data                = []
    snippet_amount_dict = {}

    # =============================================================================
    # a_combo = [file_name, its_label]
    for a_combo in combo:

        # =============================================================================
        original_file_name = a_combo[0]

        # =============================================================================
        sub_folder   = a_combo[1] + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        MFCCs_folder = a_combo[0] + "_MFCCs_block" + str(fft_length) + "_hop" + str(fft_hop) 
        MFCCs_path   = dataset_path + "/" + sub_folder + "/" + MFCCs_folder

        # =============================================================================
        MFCCs_name_list  = []
        for (dirpath, dirnames, filenames) in walk(MFCCs_path):
            MFCCs_name_list.extend(filenames)
            break
        
        # =============================================================================
        snippet_amount     = 0
        for MFCCs_name in MFCCs_name_list:
            MFCCs          = np.loadtxt(MFCCs_path + "/" + MFCCs_name)
            snippet_amount = snippet_amount + 1
            data.append([original_file_name, MFCCs_name.split('_')[1], MFCCs])
    

    print(len(data))
        
    # =============================================================================
    data_file_name = "MFCCs_"       + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop) 
    temp_file      = open(dataset_path + '/' + data_file_name + '.pickle', 'wb')
    pickle.dump(data, temp_file)
    print('data is saved')