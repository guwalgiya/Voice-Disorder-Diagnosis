# ===============================================
# Add VGGish's path
import sys
sys.path.append('../VGGish_Original')


# ===============================================
# Import Packages and Functions
from   vggish_input import wavfile_to_examples   as WavEx
from   os           import walk, path, makedirs
import numpy        as     np
import librosa


# ===============================================
# Main Function
def getVGGishInput(dataset_path, all_combo, snippet_length, snippet_hop, slash, work_on_augmented, copy_it_self):

    
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
        snippet_path = dataset_path + slash + temp_folder + slash + cur_name
        

        # ===============================================
        if copy_it_self:
            save_path = snippet_path + "_VGGish_" + "Copy"
        else:
            save_path = snippet_path + "_VGGish_" + "zeroPad"


        # ===============================================
        # Make sure the saving directory is valid
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
            S = WavEx(a_snippet_path)
            

            # ===============================================
            S_full    = np.zeros((3, S.shape[1], S.shape[2]))
            S_full[0] = S

            if copy_it_self:
               S_full[1] = S
               S_full[2] = S


            # =============================================== 
            # save that aggregated MFCC Vector in a txt version, need to compress them to a pickle file afterward
            # remark: filename has a .wav extention, remove it first
            np.savetxt(save_path + slash +  a_snippet_name[0 : -4] + '.txt', S_full, fmt = "%10.5f")
        
        
        # ===============================================
        # Report: for one file, all of its snippets' aggregated mel-spectrogram are computed and stored
        print(cur_name, "'s mel-spectrograms are computed and saved")
    

        



