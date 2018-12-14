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
        # Work_on_augmented is reserved for future work
    
  
        # ===============================================
        # Load all snippets from that file
        cur_name     = a_combo[0]
        cur_class    = a_combo[1]
        temp_folder  = cur_class    + "_"   + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        snippet_path = dataset_path + slash + temp_folder         + slash + cur_name
        

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
            L         = L / max(L)
            

            # =============================================== 
            # save that histogram in a txt version, need to compress them to a pickle file afterward
            # remark: filename has a .wav extention, remove it first
            np.savetxt(save_path + slash +  a_snippet_name[0 : -4] + ".txt", L, fmt = '%10.5f')            

        
        # ===============================================
        # Report: for one file, all of its snippets' histograms are computed and stored
        print(a_combo[0], "'s Histogram is done")

