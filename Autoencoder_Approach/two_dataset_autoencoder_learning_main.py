import math
import getCombination
import dataSplit
import loadMelSpectrogram
import autoencoder
import mySVM
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# =============================================================================
# Dataset Initialization
classes = ["Normal", "Pathol"]
dataset_main_path_train  = "/home/hguan/7100-Master-Project/Dataset-Spanish"
dataset_main_path_test   = "/home/hguan/7100-Master-Project/Dataset-KeyPentax"
train_percent            = 90
validate_percent         = 10
test_percent             = 0
# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500  #in milliseconds
snippet_hop         = 100 #in ms
mel_length          = 128
bin_size            = 500
block_size          = 512
hop_size            = 128
package             = [snippet_length, snippet_hop, block_size, hop_size, mel_length]
input_vector_length = mel_length * math.ceil(snippet_length / 1000 * fs / hop_size)
# =============================================================================
# Autoencoder Initialization
encoding_dimension = 64
encoder_layer      = 3
decoder_layer      = 3
num_epochs_auto    = 50
batch_auto         = 256
shuffle_choice     = True
loss_function      = 'mean_squared_error'
arch_bundle        = [encoder_layer, encoding_dimension, decoder_layer]
train_bundle_auto  = [num_epochs_auto, batch_auto, shuffle_choice, loss_function]
# =============================================================================
# Get
name_class_combo_train = getCombination.main(dataset_main_path_train, classes)
name_class_combo_test  = getCombination.main(dataset_main_path_test,  classes)
# =============================================================================
[train_combo, validate_combo, test_combo]  = dataSplit.main(name_class_combo_train, train_percent, validate_percent, test_percent)
[_,  _,              test_combo]           = dataSplit.main(name_class_combo_test,  0,             0,                100)
# =============================================================================
train_package     = loadMelSpectrogram.main(train_combo,    'training',   classes, package, fs, dataset_main_path_train, input_vector_length)   
validate_package  = loadMelSpectrogram.main(validate_combo, 'validating', classes, package, fs, dataset_main_path_train, input_vector_length)   
test_package      = loadMelSpectrogram.main(test_combo,     'testing',    classes, package, fs, dataset_main_path_test,  input_vector_length)
# =============================================================================
encoder = autoencoder.main(input_vector_length, train_data, validate_data, arch_bundle, train_bundle_auto)
#==============================================================================
train_data_encoded     = encoder.predict(train_data)
validate_data_encoded  = encoder.predict(validate_data)
test_data_encoded      = encoder.predict(test_data)


file_acc, snippet_acc, file_con_mat, snippet_con_mat  = mySVM.method1(train_data_encoded,    train_label2, 
                                           validate_data_encoded, validate_label2, 
                                           test_data_encoded,     test_label2,
                                           test_combo,            test_augment_amount,
                                           validate_combo,        validate_augment_amount)
    
print('--------------------------------')
print('file results')
print(file_acc)
print('--------------------------------')
print('snippet results')
print(snippet_acc)
print('--------------------------------')
print('final file results')
print(file_con_mat)
acc = 0;
for i in range(len(file_con_mat[0])):
    acc = acc + file_con_mat[i][i] / sum(file_con_mat[i])
print(acc / 2)
print('--------------------------------')
print('final snippet results')
print(snippet_con_mat)
acc = 0;
for i in range(len(snippet_con_mat[0])):
    acc = acc + snippet_con_mat[i][i] / sum(snippet_con_mat[i])
print(acc / 2)    