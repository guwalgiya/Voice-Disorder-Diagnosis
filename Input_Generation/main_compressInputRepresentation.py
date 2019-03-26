# ===============================================
# Import Packages and Functions
from compressMelSpectrogram import compressMelSpectrogram
from compressVGGishInput    import compressVGGishInput
from compressDictionary     import compressDictionary
from getCombination         import getCombination
from compressMFCCs          import compressMFCCs


# ===============================================
# Environment
parent_path = "/home/hguan/Voice-Disorder-Diagnosis/Dataset-"
slash       = "/"


# ===============================================
# Dataset Initialization, dataset = Spanish or KayPentax
classes             = ["Normal", "Pathol"]
dataset_name        = "KayPentax"
dataset_path        = parent_path + dataset_name
work_on_augmentated = True


# ===============================================
# Dsp Initialization, snippet_length, snippet_hop are in milliseconds
snippet_length = 1000  
snippet_hop    = 100 
fft_length     = 512
fft_hop        = 128
mel_length     = 128
dsp_package    = [snippet_length, snippet_hop, fft_length, fft_hop, mel_length]


# ===============================================
all_combo = getCombination(dataset_path, classes, slash)


# ===============================================
# This Line is left to be modified depends on what we need
# compressMelSpectrogram(dataset_path, classes, dsp_package, all_combo, slash)
# compressVGGishInput(dataset_path,    classes, dsp_package, all_combo, slash)
# compressDictionary(dataset_path,     classes, dsp_package, all_combo, slash, work_on_augmentated)
# compressMFCCs(dataset_path,          classes, dsp_package, all_combo, slash)