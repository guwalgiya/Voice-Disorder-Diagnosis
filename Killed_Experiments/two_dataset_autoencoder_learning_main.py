# =============================================================================
# Import Packages
import matplotlib
matplotlib.use('Agg')
import getCombination
from   keras.models              import load_model
from   loadMelSpectrogram        import loadMelSpectrogram
from   keras.models              import Model
import numpy                     as     np
import tensorflow                as     tf
import math
import dataSplit
import autoencoder
import mySVM
import pickle
import os

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
sess        = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =============================================================================
# Dataset Initialization
classes = ["Normal", "Pathol"]
dataset_path_train  = "/home/hguan/7100-Master-Project/Dataset-KayPentax"
dataset_path_test   = "/home/hguan/7100-Master-Project/Dataset-Spanish"
train_percent            = 90
validate_percent         = 10
test_percent             = 0
best_model_name    = "best_model_this_fold.hdf5"

# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500   #in milliseconds
snippet_hop         = 100   #in ms
fft_length          = 512
fft_hop             = 128
mel_length          = 128
num_MFCCs           = 20
num_rows            = num_MFCCs
dsp_package         = [fs, snippet_length, snippet_hop, fft_length, fft_hop]
input_vector_length = num_rows * math.ceil(snippet_length / 1000 * fs / fft_hop)
input_name          = "MelSpectrogram"

# =============================================================================
# Autoencoder Initialization
encoding_dimension = 2
encoder_layer      = 9
decoder_layer      = 9
epoch_limit        = 1000000
batch_auto         = 1024
shuffle_choice     = True
loss_function      = 'mean_squared_error'
arch_bundle        = [encoder_layer, encoding_dimension, decoder_layer]
train_bundle_auto  = [epoch_limit, batch_auto, shuffle_choice, loss_function]

# =============================================================================
data_file_name       = input_name + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
aug_dict_file_name   = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"                   
unaug_dict_file_name = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" 

temp_file_1          = open(dataset_path_train + '/' + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path_train + '/' + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path_train + '/' + unaug_dict_file_name + '.pickle', 'rb')
data_1               = pickle.load(temp_file_1)
aug_dict_1           = pickle.load(temp_file_2)
unaug_dict_1         = pickle.load(temp_file_3)

temp_file_1          = open(dataset_path_test + '/' + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path_test + '/' + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path_test + '/' + unaug_dict_file_name + '.pickle', 'rb')
data_2               = pickle.load(temp_file_1)
aug_dict_2           = pickle.load(temp_file_2)
unaug_dict_2         = pickle.load(temp_file_3)


# =============================================================================
train_validate_combo    = getCombination.main(dataset_path_train, classes)
normal_name_class_combo = [combo for combo in train_validate_combo if combo[1] == "Normal"]
pathol_name_class_combo = [combo for combo in train_validate_combo if combo[1] == "Pathol"]

# =============================================================================
[normal_train_combo, normal_validate_combo, _] = dataSplit.main(normal_name_class_combo, 90, 10, 0)
[pathol_train_combo, pathol_validate_combo, _] = dataSplit.main(pathol_name_class_combo, 90, 10, 0)

# =============================================================================
train_combo    = normal_train_combo    + pathol_train_combo    
validate_combo = normal_validate_combo + pathol_validate_combo

# =============================================================================
test_combo          = getCombination.main(dataset_path_test,  classes)
[_, _, test_combo]  = dataSplit.main(test_combo, 0, 0, 100)


# =============================================================================
train_package     = loadMelSpectrogram(train_combo,    classes, dsp_package, num_rows, "MFCCs", data_1, True,  aug_dict_1)   
validate_package  = loadMelSpectrogram(validate_combo, classes, dsp_package, num_rows, "MFCCs", data_1, False, unaug_dict_1)   
test_package      = loadMelSpectrogram(test_combo,     classes, dsp_package, num_rows, "MFCCs", data_2, False, unaug_dict_2)

# =============================================================================
train_data,    _,  _, train_label_3,    train_dist,    _                       = train_package
validate_data, _,  _, validate_label_3, validate_dist, validate_augment_amount = validate_package
test_data,     _,  _, test_label_3,     test_dist,     test_augment_amount     = test_package
print(train_dist)
print(validate_dist)
print(test_dist)

for i in range(num_rows):
    max_standard           = max(np.amax(train_data[:, i, :]), np.amax(validate_data[:, i, :]))
    min_standard           = min(np.amin(train_data[:, i, :]), np.amin(validate_data[:, i, :]))
    mean                   = np.mean(train_data[:, i, :].flatten().tolist() + validate_data[:, i, :].flatten().tolist())
    std                    = np.std(train_data[:, i, :].flatten().tolist()  + validate_data[:, i, :].flatten().tolist())
    train_data[:, i, :]    = (train_data[:, i, :]    - mean) / std
    validate_data[:, i, :] = (validate_data[:, i, :] - mean) / std
    test_data[:, i, :]     = (test_data[:, i, :]     - mean) / std


# =============================================================================
train_data    = train_data.reshape((len(train_data),       np.prod(train_data.shape[1:])),    order = 'F') 
validate_data = validate_data.reshape((len(validate_data), np.prod(validate_data.shape[1:])), order = 'F')
test_data     = test_data.reshape((len(test_data),         np.prod(test_data.shape[1:])),     order = 'F')


# =============================================================================
_, history, encodeLayer_index = autoencoder.main(input_vector_length, train_data, test_data, arch_bundle , train_bundle_auto)
best_autoencoder              = load_model(best_model_name)


# ===============================================================================
# Find the best encoding dimension    
best_dim         = 0
best_file_acc    = 0
possible_encoder = []
best_result_pack = None
print("Encoder is trained")
for dim in [2, 4, 8, 16, 32, 64, 128, 256,512, 1024]:
    
    # ===============================================================================
    index        = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].index(dim)
    best_encoder = Model(inputs  = best_autoencoder.inputs, outputs = best_autoencoder.layers[encodeLayer_index - index].output)
    
    # ===============================================================================
    train_data_encoded     = best_encoder.predict(train_data)
    validate_data_encoded  = best_encoder.predict(validate_data)
    test_data_encoded      = best_encoder.predict(test_data)


    # ===============================================================================
    fold_result_package  = mySVM.method1(train_data_encoded,    train_label_3, 
                                         validate_data_encoded, validate_label_3, 
                                         test_data_encoded,     test_label_3,
                                         test_combo,            test_augment_amount)
    
    # =============================================================================
    file_acc, file_con_mat, snippet_acc, snippet_con_mat = fold_result_package
    print(dim, file_acc)
    if file_acc > best_file_acc:
        best_dim         = dim
        best_file_acc    = file_acc
        best_result_pack = fold_result_package

file_acc, file_con_mat, snippet_acc, snippet_con_mat = best_result_pack


# =============================================================================
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