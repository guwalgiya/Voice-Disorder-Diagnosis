# =============================================================================
# Import Packages
import librosa
import math
import numpy as np
from os import walk
from os import path
from os import makedirs


# =============================================================================
fs     = 25000
new_fs = 16000
cut_length = 500 #must be in mili-seconds
hopsizeInms = 100 #must be in mili-seconds
parent_folder = "/home/hguan/7100-Master-Project/Dataset-Spanish"
vocal_types = ["Normal", "Pathol"]

# =============================================================================


def main(parent_folder, vocal_type, fs, new_fs, cut_length, hopSizeInms):
    
    input_path         =  parent_folder + "/" + vocal_type + "_Pitch_Shifted"
    output_main_folder =  parent_folder + "/" + vocal_type + "_" + str(cut_length) + "ms_" + str(hopSizeInms) + "ms"
    
    dir = path.dirname(output_main_folder + "/dummy.aaa")
    if not path.exists(dir):
        makedirs(dir)
    
    filename = []
    for (dirpath, dirnames, filenames) in walk(input_path):
        filename.extend(filenames)
        break

    for name in filename:
        x, _= librosa.load(input_path + "/" + name, sr = fs)
        x = librosa.resample(x, fs, new_fs)
        
        blockSize = math.floor(new_fs * cut_length  / 1000)
        hopSize   = math.floor(new_fs * hopSizeInms / 1000)
        
        #print(name)
        blocking(x, blockSize, hopSize, new_fs, output_main_folder, name[0 : len(name) - 4] + "_" + str(cut_length)  + "ms"
                                                                                                 + "_" + str(hopSizeInms) + "ms")        
        print(name, "is cut into snippets")
    


def blocking(x, blockSize, hopSize, new_fs, output_main_folder, filename):
    xb = block_audio(x, blockSize, hopSize)
    _, n = xb.shape
    #print(n)
    for i in range(n):
        x_i = xb[:,i]
        if i + 1 <= 9:
            index = "0" + str(i + 1)
        else:
            index = str(i + 1)
        
        filename_parts = filename.split("_")
        main_file_name = filename_parts[0] #for example: aala, aasa
        output_sub_folder = output_main_folder + "/" + main_file_name
        dir = path.dirname(output_sub_folder + "/dummy.aaa")
        if not path.exists(dir):
            makedirs(dir)
        librosa.output.write_wav(output_sub_folder + "/" + filename + "_" + index +  ".wav", x_i, new_fs) 


def block_audio(x, blockSize, hopSize):
    num_blocks = math.ceil( len(x) / hopSize);
    #print(num_blocks)
    xb = np.zeros( (blockSize, num_blocks) );
    for i in range(num_blocks):
        try:
            xb[:, i] = x[i * hopSize : (i * hopSize + blockSize)]  
        except:
            i = i + 1
            break
    xb = xb[:, 0 : i - 1]
    return xb

for vocal_type in vocal_types:
    main(parent_folder, vocal_type, fs, new_fs, cut_length, hopsizeInms)
