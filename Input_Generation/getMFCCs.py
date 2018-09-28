from os import walk
from os import path
from os import makedirs
import numpy as np
import librosa

def main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, block_size, hop_size):
    j = 0
    for a_combo in name_class_combo:
        a_class = a_combo[1]
        temp_folder = a_class + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        snippet_path = dataset_main_path + "/" + temp_folder + "/" + a_combo[0]
        
        save_path = snippet_path + "_MFCCs_block" + str(block_size) + "_hop" + str(hop_size)
        dir = path.dirname(save_path + "/dummy.aaa")
        if not path.exists(dir):
            makedirs(dir)
        
        snippet_names = []
        for (dirpath, dirnames, filenames) in walk(snippet_path):
            snippet_names.extend(filenames)
            break
                     
        for a_snippet_name in snippet_names:
            a_snippet_path = snippet_path + "/" + a_snippet_name
            x, _ = librosa.load(a_snippet_path, sr = fs)
            S = librosa.feature.melspectrogram(y = x, sr = fs, n_fft = block_size, hop_length = hop_size)
            mfccs = librosa.feature.mfcc(S = librosa.power_to_db(S), n_mfcc = 20)
            aggregate_mfccs = []
            for i in range(mfccs.shape[0]):
                aggregate_mfccs.append(np.mean(mfccs[i, :]))
                aggregate_mfccs.append(np.std(mfccs[i, :]))
            np.savetxt(save_path +"/" +  a_snippet_name[0 : -4] + '.txt', aggregate_mfccs, fmt='%10.5f')
            
        j = j + 1    
        print(a_combo[0], "'s MFCCs are done", j)


