# ===============================================
# Import Packages and Functions
from   os       import walk, path, makedirs
import numpy    as np
import librosa
import math


# ===============================================
# Environment
slash       = "/"
parent_path = "/home/hguan/7100-Master-Project/Dataset-"


# ===============================================
# Dataset Initialization
classes             = ["Normal", "Pathol"]
dataset_name        = "Spanish"
dataset_path        = parent_path + dataset_name
work_on_augmentated = True


# ===============================================
# Dsp Initialization, snippet_length, snippet_hop are in milliseconds
fs              = 25000
new_fs          = 16000
snippet_hop     = 100 
snippet_length  = 500 


# ===============================================
# Main Function for this script
def cut_audio_file_to_snippets(dataset_path, a_class, fs, new_fs, snippet_length, snippet_hop):


    # ===============================================
    if work_on_augmentated:
        input_path         =  dataset_path + slash + a_class + "_Pitch_Shifted"
        output_main_folder =  dataset_path + slash + a_class + "_"              + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
    else:
        input_path         =  dataset_path + slash + a_class
        output_main_folder =  dataset_path + slash + a_class + "_"              + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented"
    

    # ===============================================
    dir = path.dirname(output_main_folder + "/dummy.aaa")
    if not path.exists(dir):
        makedirs(dir)
    

    # ===============================================
    filename = []
    for (dirpath, dirnames, filenames) in walk(input_path):
        filename.extend(filenames)
        break
    

    # ===============================================
    for name in filename:
        x, _ = librosa.load(input_path + slash + name, sr = fs)
        x    = librosa.resample(x, fs, new_fs)
        

        # ===============================================
        fft_length = math.floor(new_fs * snippet_length  / 1000)
        fft_hop    = math.floor(new_fs * snippet_hop     / 1000)
        
        
        # ===============================================
        blocking(x, fft_length, fft_hop, new_fs, output_main_folder, name[0 : len(name) - 4] + "_" + str(snippet_length)  + "ms"
                                                                                             + "_" + str(snippet_hop)     + "ms")        
        

        # ===============================================
        print(name, "is cut into snippets")
    

# ===============================================
def blocking(x, fft_length, fft_hop, new_fs, output_main_folder, filename):


	# ===============================================
    xb = block_audio(x, fft_length, fft_hop)
    

    # ===============================================
    _, n = xb.shape


    # ===============================================
    for i in range(n):
        x_i = xb[:,i]
        if i + 1 <= 9:
            index = "0" + str(i + 1)
        else:
            index = str(i + 1)
        

        # ===============================================
        filename_parts    = filename.split("_")
        main_file_name    = filename_parts[0] 
        output_sub_folder = output_main_folder + slash + main_file_name
        

        # ===============================================
        dir = path.dirname(output_sub_folder + "/dummy.aaa")
        if not path.exists(dir):
            makedirs(dir)


        # ===============================================
        librosa.output.write_wav(output_sub_folder + slash + filename + "_" + index + ".wav", x_i, new_fs) 


# ===============================================
# Function
def block_audio(x, fft_length, fft_hop):


	# ===============================================
    num_blocks = math.ceil(len(x) / fft_hop)


    # ===============================================
    xb = np.zeros((fft_length, num_blocks))


    # ===============================================
    for i in range(num_blocks):
        try:
            xb[:, i] = x[i * fft_hop : (i * fft_hop + fft_length)]  
        except:
            i = i + 1
            break


    # ===============================================
    xb = xb[:, 0 : i - 1]


    # ===============================================
    return xb


# ===============================================
# Run this script, this script should be run after augmentation_pitch_shift
for a_class in classes:
    cut_audio_file_to_snippets(dataset_path, a_class, fs, new_fs, snippet_length, snippet_hop)
