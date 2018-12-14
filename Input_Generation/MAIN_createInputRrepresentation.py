# ===============================================
# Import Packages and Functions
from getMFCCs             import getMFCCs
from getCombination       import getCombination
from getMelSpectrogram    import getMelSpectrogram
from getLoudnessHistogram import getLoudnessHistogram


# ===============================================
# Environment
parent_path = "/home/hguan/7100-Master-Project/Dataset-"
slash       = "/"


# ===============================================
# Dataset Initialization, dataset = Spanish or KayPentax
classes             = ["Normal", "Pathol"]
dataset_name        = "Spanish"
dataset_path        = parent_path + dataset_name
work_on_augmentated = False


# ===============================================
# Dsp Initialization, snippet_length, snippet_hop are in milliseconds
snippet_length = 500   
snippet_hop    = 100 
fft_length     = 512
fft_hop        = 128
fs             = 16000


# ===============================================
# Factor for Loudness Histogram, no strict requirement on bin_size choice
bin_size  = 1000


# ===============================================
# Factor for MelSpectrogram
mel_length = 128


# ===============================================
all_combo = getCombination(dataset_path, classes, slash)



# ===============================================
# Run functions
getMFCCs(dataset_path,             all_combo, fs, snippet_length, snippet_hop, slash, work_on_augmented, fft_length, fft_hop)
getMelSpectrogram(dataset_path,    all_combo, fs, snippet_length, snippet_hop, slash, work_on_augmented, fft_length, fft_hop, mel_length)
getLoudnessHistogram(dataset_path, all_combo, fs, snippet_length, snippet_hop, slash, work_on_augmented, bin_size)


