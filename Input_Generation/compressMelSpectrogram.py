from   os      import walk
import numpy   as     np
import librosa
import pickle


# =============================================================================
# Main function of this file
def compressMelSpectrogram(dataset_path, classes, dsp_package, augmented, combo):
    
    # =============================================================================
    snippet_length, snippet_hop, fft_length, fft_hop, mel_length = dsp_package
    
    # =============================================================================
    data                = []
    snippet_amount_dict = {}

    # =============================================================================
    # a_combo = [file_name, its_label]
    for a_combo in combo:

        # =============================================================================
        # label_1: 1, label_2:[0,1], label_3: pathol
        original_file_name    = a_combo[0]
        
        class_index           = classes.index(a_combo[1])
        label_1               = class_index

        label_2               = [0] * len(classes)
        label_2[class_index]  = 1          
        
        label_3               = a_combo[1]
        

        if original_file_name in ["apa_p", "araa_p", "arba_p", "cpca_p", "cpra_p", "fgaa_p", "jaaa_p", "jgsa_p", "jmca_p"]:
            a_combo[0] = a_combo[0][0 : -2]

        # =============================================================================
        if augmented:
            sub_folder = a_combo[1] + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        else:
            sub_folder = a_combo[1] + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented"

        # =============================================================================
        spectrogram_folder = a_combo[0] + "_MelSpectrogram_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
        spectrogram_path   = dataset_path + "/" + sub_folder + "/" + spectrogram_folder

        # =============================================================================
        spectrogram_name_list  = []
        for (dirpath, dirnames, filenames) in walk(spectrogram_path):
            spectrogram_name_list.extend(filenames)
            break
        
        # =============================================================================
        snippet_amount     = 0
        for spectrogram_name in spectrogram_name_list:
            # melspectrogram = np.loadtxt(spectrogram_path + "/" + spectrogram_name)
            snippet_amount = snippet_amount + 1
            # if (a_combo[0] in ["apa", "araa", "arba", "cpca", "cpra", "fgaa", "jaaa", "jgsa", "jmca"]) and (a_combo[1] == "Pathol"):
            #     data.append([original_file_name + "_P", spectrogram_name.split('_')[1], melspectrogram])
            # else:
            #     data.append([original_file_name,        spectrogram_name.split('_')[1], melspectrogram])


        if (a_combo[0] in ["apa", "araa", "arba", "cpca", "cpra", "fgaa", "jaaa", "jgsa", "jmca"]) and (a_combo[1] == "Pathol"):      
            snippet_amount_dict[original_file_name] = [snippet_amount, label_1, label_2, label_3]
        else:
            snippet_amount_dict[original_file_name] = [snippet_amount, label_1, label_2, label_3]
        print(original_file_name, label_1, snippet_amount)
    

    print(len(data))
        
    # =============================================================================
    if augmented:
        #data_file_name = "MelSpectrogram_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" +                  "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
        dict_file_name = "Dictionary_"     + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" +                  "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
        #temp_file_0    = open(dataset_path + '/' + data_file_name + '.pickle', 'wb')
        #pickle.dump(data, temp_file_0)
        print('data is saved')

    else:
        #data_file_name = "MelSpectrogram_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
        dict_file_name = "Dictionary_"     + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
    
    with open(dataset_path + '/' + dict_file_name + '.pickle', 'wb') as temp_file_1:
        pickle.dump(snippet_amount_dict, temp_file_1, protocol = pickle.HIGHEST_PROTOCOL)
    print('Dictionary is saved')