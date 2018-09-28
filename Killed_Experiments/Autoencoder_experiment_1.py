from keras.layers import Input, Dense
from keras.models import Model
import math
# =============================================================================
# Dataset Initialization
dataset_main_path = "C:\\Music Technology Master\\7100 - Master Project\\Dataset - Spanish"
vocal_types_train = {"Normal" : 0, "Pathol" : 0} 
vocal_types_test = {"Normal" : 0, "Pathol" : 0} 
# =============================================================================
# =============================================================================
# Dsp Initialization
snippet_length = 500  #in milliseconds
fs = 16000
block_size = 512
hop_size = 256
mel_length = 128
# =============================================================================

# =============================================================================
# Autoencoder Initialization
encoding_dimension = 64
num_epochs = 50
batch = 256
shuffle_choice = True
# =============================================================================

input_vector_length = mel_length * math.ceil(snippet_length / 1000 * fs / hop_size)

input_mel_spectrogram = Input(shape = (input_vector_length, ))

encoded = Dense(encoding_dimension, activation = 'relu')(input_mel_spectrogram)

decoded = Dense(input_vector_length, activation = 'sigmoid')(encoded)

autoencoder = Model(input_mel_spectrogram, decoded)
encoder = Model(input_mel_spectrogram, encoded)

encoded_input = Input(shape = (encoding_dimension, ) )

decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')

import numpy as np
from os import walk
import librosa

def load_data(vocal_types, dataset_main_path, train_or_test, snippet_length):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    file_names = []
    for vocal_type in list(vocal_types.keys()):
        print("Now searching " + vocal_type + " recordings for " + train_or_test + "ing.")
        input_path = dataset_main_path + "\\" + vocal_type + "_Augmented_" + str(snippet_length) + "ms_" + train_or_test
    
        for (dirpath, dirnames, filenames) in walk(input_path):
            file_names.extend(filenames)
            vocal_types[vocal_type] = len(filenames)
            print('We found ' + str(len(filenames)) + ' files')
            print('----------------------------------------------------')
            break
       
    x_loaded = np.zeros( (len(file_names), mel_length, int(input_vector_length / mel_length)))
    
    for vocal_type in list(vocal_types.keys()):
        print("Now getting mel spectrogrm from " + vocal_type + " recordings for " + train_or_test + "ing.")
        input_path = dataset_main_path + "\\" + vocal_type + "_Augmented_" + str(snippet_length) + "ms_" + train_or_test
        
        i = 0
        for filename in file_names:
            x, _ = librosa.load(input_path + "\\" + filename, sr = fs)
            S = librosa.feature.melspectrogram(y = x, sr = fs, n_fft = block_size, hop_length = hop_size)
            x_loaded[i] = S / S.max()
            i = i + 1
            if i == vocal_types[vocal_type] - 1:
               file_names = file_names[ vocal_types[vocal_type] : ]
               break
        
    return x_loaded


x_train = load_data(vocal_types_train, dataset_main_path, 'train', snippet_length)   
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
print('Training Data is Loaded!')

x_test  = load_data(vocal_types_test, dataset_main_path, 'test',  snippet_length)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print('Testing Data is Loaded!')

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Autoencoder Starts')
autoencoder.fit(x_train, x_train,
                epochs = num_epochs,
                batch_size = batch,
                shuffle = shuffle_choice,
                validation_data = (x_test, x_test)
                )

