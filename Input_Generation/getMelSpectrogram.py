from os import walk
from os import path
from os import makedirs
import numpy as np
import librosa

def main(dataset_main_path, name_class_combo, fs, snippet_length, snippet_hop, block_size, hop_size, mel_length):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') 
    j = 0
    for a_combo in name_class_combo:
        a_class = a_combo[1]
        temp_folder = a_class + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        snippet_path = dataset_main_path + "/" + temp_folder + "/" + a_combo[0]
        
        save_path = snippet_path + "_MelSpectrogram_block" + str(block_size) + "_hop" + str(hop_size) + "_mel" + str(mel_length) 
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
            S = librosa.feature.melspectrogram(y = x, sr = fs, n_fft = block_size, hop_length = hop_size, n_mels = mel_length)
            #S = S / S.max()
            np.savetxt(save_path +"/" +  a_snippet_name[0:-4] + '.txt', S, fmt='%10.5f')
            
        j = j + 1
        print(a_combo[0], "'s spectrogram is done", j)



