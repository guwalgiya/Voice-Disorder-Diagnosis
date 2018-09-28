from os import walk
from os import path
from os import makedirs
import numpy as np
import librosa

def main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, bin_size, slash):
    #note: linux slash = /, windows = \\
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') 
    j = 0
    for a_combo in name_class_combo:
        a_class      = a_combo[1]
        temp_folder  = a_class + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        snippet_path = dataset_main_path + slash + temp_folder + slash + a_combo[0]
        
        save_path = snippet_path + "_LoudHistogram_bin" + str(bin_size)
        dir = path.dirname(save_path + slash + "dummy.aaa")
        if not path.exists(dir):
            makedirs(dir)
        
        snippet_names = []
        for (dirpath, dirnames, filenames) in walk(snippet_path):
            snippet_names.extend(filenames)
            break
                     
        for a_snippet_name in snippet_names:
            a_snippet_path = snippet_path + slash + a_snippet_name
            x, _           = librosa.load(a_snippet_path, sr = fs)            
            bin_edges      = np.linspace(0, 1, bin_size + 1, endpoint = True)            
            L,_            = np.histogram(abs(x), bins = bin_edges)            
            #normalization
            L              = L / max(L)
            
            np.savetxt(save_path + slash +  a_snippet_name[0:-4] + '.txt', L, fmt='%10.5f')            
            j = j + 1
            
        print(a_combo[0], "'s Histogram is done")

