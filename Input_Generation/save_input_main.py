from   compressMelSpectrogram import compressMelSpectrogram
from   compressMFCCs          import compressMFCCs
import numpy                  as np 
import getCombination

# =============================================================================
# Dataset Initialization
classes        = ["Normal", "Pathol"]
dataset_name   = "Spanish"
dataset_path   = "/home/hguan/7100-Master-Project/Dataset-" + dataset_name
augmented      = False

# =============================================================================
# Dsp Initialization
snippet_length = 500   #in milliseconds
snippet_hop    = 100   #in ms
fft_length     = 512
fft_hop        = 128
mel_length     = 128
dsp_package    = [snippet_length, snippet_hop, fft_length, fft_hop, mel_length]


# =============================================================================
# Load saved features into a "pickle" file, True for augmented file, o/w False
name_class_combo          = getCombination.main(dataset_path, classes, "/")

#compressMelSpectrogram(dataset_path, classes, dsp_package, True,  name_class_combo)
compressMelSpectrogram(dataset_path, classes, dsp_package, False, name_class_combo)
#compressMelSpectrogram(dataset_path, classes, dsp_package, augmented, name_class_combo)
#compressMFCCs(dataset_path, classes, dsp_package, name_class_combo)