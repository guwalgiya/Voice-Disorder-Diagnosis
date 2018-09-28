from keras.layers import Input, Dense
from keras.models import Model
import math
import random
# =============================================================================
# Dataset Initialization
dataset_main_path = "C:\\Music Technology Master\\7100 - Master Project\\Dataset - Spanish"
train_percent     = 70
validate_percent  = 30
test_percentage   = 0
classes = ['Normal','Pathol']
# =============================================================================
# =============================================================================
# Dsp Initialization
snippet_length = 700  #in milliseconds
snippet_hop = 200 #in ms
fs = 16000
block_size = 512
hop_size = 256
mel_length = 128
# =============================================================================

# =============================================================================
# Autoencoder Initialization
encoding_dimension = 32
encoder_layer = 3
decoder_layer = 3
num_epochs = 50
batch = 256
shuffle_choice = True
loss_function = 'mean_squared_error'
# =============================================================================


# model setup
input_vector_length = mel_length * math.ceil(snippet_length / 1000 * fs / hop_size)

input_mel_spectrogram = Input(shape = (input_vector_length, ))

encoded = Dense(encoding_dimension * 2, activation = 'relu')(input_mel_spectrogram)
encoded = Dense(encoding_dimension,     activation = 'relu')(encoded)

decoded = Dense(encoding_dimension * 2, activation = 'relu')(encoded)
decoded = Dense(input_vector_length, activation = 'sigmoid')(decoded)

autoencoder = Model(input_mel_spectrogram, decoded)
encoder = Model(input_mel_spectrogram, encoded)

# =============================================================================
# encoded_input = Input(shape = (encoding_dimension, ) )
# 
# decoder_layer = autoencoder.layers[-1]
# 
# decoder = Model(encoded_input, decoder_layer(encoded_input))
# =============================================================================

autoencoder.compile(optimizer = 'adadelta', loss = loss_function)



# load data
import numpy as np
from os import walk
import librosa

def load_data(name_class_combo):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    x_label = []   
    x_distribution = {}
    for a_class in classes:
        x_distribution[a_class] = 0
    
    whole_path = []    
    for main_name in name_class_combo:
        a_class = main_name[1]
        temp_folder = a_class + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
        temp_path = dataset_main_path + "\\" + temp_folder + "\\" + main_name[0]
        
        sub_names = []
        for (dirpath, dirnames, filenames) in walk(temp_path):
            sub_names.extend(filenames)
            break
        
        x_label = x_label + [a_class] * len(sub_names)
        x_distribution[a_class] = x_distribution[a_class] + 1
        
        for i in range(len(sub_names)):
            whole_path.append(temp_path + "\\" + sub_names[i])
                
    x_loaded = np.zeros((len(x_label), mel_length, int(input_vector_length / mel_length)))
    i = 0
    for a_path in whole_path:
        x, _ = librosa.load(a_path, sr = fs)
        S = librosa.feature.melspectrogram(y = x, sr = fs, n_fft = block_size, hop_length = hop_size, n_mels = mel_length)
        x_loaded[i] = S / S.max()
        i = i + 1
        if i % 1000 == 0:
            print(i,'/',len(whole_path),'is done' )
    return x_loaded, x_label, x_distribution


name_class_combo = []
for a_class in classes:
    names = []
    for (dirpath, dirnames, filenames) in walk(dataset_main_path + "\\" + a_class):
        names.extend(filenames)
        break
    for i in range(len(names)):
        names[i] =  names[i][0 : len(names[i]) - 4]       
        name_class_combo.append([names[i], a_class])

     

#split trainining, validating and testing
random.shuffle(name_class_combo)    
#train_combo    = name_class_combo[:round(train_percent / 100 * len(name_class_combo))]    
#validate_combo = name_class_combo[round(train_percent / 100 * len(name_class_combo)) :]    
#test_names = ...
 
#load training data
x_train, x_train_label, x_train_dist = load_data(train_combo)   
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
print('Training Data is Loaded!')

#load testing data
x_validate, x_validate_label, x_validate_dist = load_data(validate_combo)   
x_validate = x_validate.reshape((len(x_validate), np.prod(x_validate.shape[1:])))
print('Validating Data is Loaded!')

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Autoencoder Starts')
autoencoder.fit(x_train, x_train,
                epochs = num_epochs,
                batch_size = batch,
                shuffle = shuffle_choice,
                validation_data = (x_validate, x_validate)
                )
