# ===============================================
# Import Packages and Functions
from   os      import walk
import numpy   as     np
import pickle


# ===============================================
# MainFunction
def compressMelSpectrogram(dataset_path, classes, dsp_package, all_combo, slash):
  

    # ===============================================
    snippet_length, snippet_hop, fft_length, fft_hop, mel_length = dsp_package
    

    # ===============================================
    data = []


    # ===============================================
    # a_combo = [file_name, its_label], example: "[aaa, "Pathol"]"
    for a_combo in all_combo:


        # ===============================================
        original_file_name = a_combo[0]


        # ===============================================  
        if original_file_name in ["apa_p", "araa_p", "arba_p", "cpca_p", "cpra_p", "fgaa_p", "jaaa_p", "jgsa_p", "jmca_p"]:
            a_combo[0] = a_combo[0][0 : -2]

   
        # ===============================================
        sub_folder         = a_combo[1]   + "_"                     + str(snippet_length) + "ms_"  + str(snippet_hop)   + "ms"
        spectrogram_folder = a_combo[0]   + "_MelSpectrogram_block" + str(fft_length)     + "_hop" + str(fft_hop)       + "_mel" + str(mel_length)
        spectrogram_path   = dataset_path + slash                   + sub_folder          + slash  + spectrogram_folder


        # ===============================================
        spectrogram_name_list = []
        for (dirpath, dirnames, filenames) in walk(spectrogram_path):
            spectrogram_name_list.extend(filenames)
            break
        

        # ===============================================
        snippet_amount = 0
        for spectrogram_name in spectrogram_name_list:
            
            
            # ===============================================
            mel_spectrogram = np.loadtxt(spectrogram_path + slash + spectrogram_name)
            

            # ===============================================
            snippet_amount = snippet_amount + 1


            # ===============================================
            # [file_name, pitch shift by semitone, mel_spectrogram] example: [aaa, "U1.5", mel_spectrogram]
            if (original_file_name in ["apa_p", "araa_p", "arba_p", "cpca_p", "cpra_p", "fgaa_p", "jaaa_p", "jgsa_p", "jmca_p"]):
                data.append([original_file_name, spectrogram_name.split('_')[1], mel_spectrogram])
            else:
                data.append([original_file_name, spectrogram_name.split('_')[1], mel_spectrogram])


    # ===============================================
    print(len(data))
        

    # ===============================================
    data_file_name = "MelSpectrogram_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)


    # ===============================================
    temp_file = open(dataset_path + slash + data_file_name + '.pickle', 'wb')
    

    # ===============================================
    pickle.dump(data, temp_file)

    
    # ===============================================
    print("MFCC data is saved as a Pickle File")

    