# ===============================================
# Import Packages and Functions
from   os      import walk, path, makedirs
import numpy   as     np
import librosa


# ===============================================
# Main Function
def getMelSpectrogram(dataset_path, all_combo, fs, snippet_length, snippet_hop, slash, work_on_augmented, fft_length, fft_hop, mel_length):

    
    # ===============================================
    for a_combo in all_combo:
        
        
        # ===============================================
        # Basic Information
        cur_name  = a_combo[0]
        cur_class = a_combo[1]
    
    
        # ===============================================
        # Load all snippets from that file
        if work_on_augmented:
            temp_folder = cur_class + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        else:
            temp_folder = cur_class + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented"
        
        
        # ===============================================
        snippet_path = dataset_path + slash + temp_folder         + slash + cur_name
        
        
        # ===============================================
        # Make sure the saving directory is valid
        save_path = snippet_path           + "_MelSpectrogram_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length) 
        dir       = path.dirname(save_path + "/dummy.aaa")
        if not path.exists(dir):
            makedirs(dir)
        
        
        # ===============================================
        snippet_names = []
        for (dirpath, dirnames, filenames) in walk(snippet_path):
            snippet_names.extend(filenames)
            break
        
        # =============================================== 
        for a_snippet_name in snippet_names:
            a_snippet_path = snippet_path + slash + a_snippet_name
            
            
            # =============================================== 
            x, _ = librosa.load(a_snippet_path,  sr = fs)
            S    = librosa.feature.melspectrogram(y = x, sr = fs, n_fft = fft_length, hop_length = fft_hop, n_mels = mel_length)
            
            
            # =============================================== 
            # save that aggregated MFCC Vector in a txt version, need to compress them to a pickle file afterward
            # remark: filename has a .wav extention, remove it first
            np.savetxt(save_path + slash +  a_snippet_name[0:-4] + '.txt', S, fmt = "%10.5f")
        
        
        # ===============================================
        # Report: for one file, all of its snippets' aggregated mel-spectrogram are computed and stored
        print(cur_name, "'s mel-spectrograms are computed and saved")
    

        



