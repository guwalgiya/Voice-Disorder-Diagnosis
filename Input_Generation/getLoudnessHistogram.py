# ===============================================
# Import Packages and Functions
from   os      import walk, path, makedirs
import numpy   as     np
import librosa


# ===============================================
# MainFunction
def getLoudnessHistogram(dataset_path, all_combo, fs, snippet_length, snippet_hop, slash, work_on_augmented, bin_size):

    
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
        # Make sure the saving directory is valid
        save_path = snippet_path           + "_LoudHistogram_bin" + str(bin_size)
        dir       = path.dirname(save_path + slash                + "dummy.aaa")
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
            # Compute Loudness histogram for a particular snippet
            bin_edges = np.linspace(0, 1, bin_size + 1, endpoint = True)
            x, _      = librosa.load(a_snippet_path,    sr       = fs)            
            L, _      = np.histogram(abs(x),            bins     = bin_edges)            
            

            # =============================================== 
            # save that histogram in a txt version, need to compress them to a pickle file afterward
            # remark: filename has a .wav extention, remove it first
            np.save(save_path + slash +  a_snippet_name[0 : -4], L)            

        
        # ===============================================
        # Report: for one file, all of its snippets' histograms are computed and stored
        print(cur_name, "'s Loudness Histograms are computed and saved")

