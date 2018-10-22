from   sklearn.model_selection import KFold
from   loadMelSpectrogram      import loadMelSpectrogram
import numpy                 as     np
import math
import getCombination
import dataSplit
import autoencoderRandomWeights
import mySVM
import pickle

# =============================================================================
# Dataset Initialization
classes        = ["Normal", "Pathol"]
dataset_name   = "KayPentax"
dataset_path   = "/home/hguan/7100-Master-Project/Dataset-" + dataset_name
num_folds      = 5   
train_percent  = 75 

# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500   #in milliseconds
snippet_hop         = 100   #in ms
fft_length          = 512
fft_hop             = 128
mel_length          = 128
dsp_package         = [fs, snippet_length, snippet_hop, fft_length, fft_hop, mel_length]
input_vector_length = mel_length * math.ceil(snippet_length / 1000 * fs / fft_hop)
vector_length       = 128
input_name          = "MelSpectrogram"

# =============================================================================
# Autoencoder Initialization
encoding_dimension = 256
encoder_layer      = 4
decoder_layer      = 4
epoch_limit        = 10000
batch_auto         = 128
shuffle_choice     = True
loss_function      = 'mean_squared_error'
arch_bundle        = [encoder_layer, encoding_dimension, decoder_layer]
train_bundle_auto  = [epoch_limit, batch_auto, shuffle_choice, loss_function]
fold_num           = 1

# =============================================================================
name_class_combo        = getCombination.main(dataset_path, classes)
normal_name_class_combo = [combo for combo in name_class_combo if combo[1] == "Normal"]
pathol_name_class_combo = [combo for combo in name_class_combo if combo[1] == "Pathol"]
normal_name_class_combo = np.asarray(normal_name_class_combo)
pathol_name_class_combo = np.asarray(pathol_name_class_combo)
num_normal              = len(normal_name_class_combo)
num_pathol              = len(pathol_name_class_combo)
print(num_normal, num_pathol)

normal_spliter      = KFold(n_splits = num_folds, shuffle = True)
normal_index_array  = np.arange(len(normal_name_class_combo))
normal_folds        = [(train_validate_index, test_index) for train_validate_index, test_index in normal_spliter.split(normal_index_array)]


pathol_spliter      = KFold(n_splits = num_folds, shuffle = True)
pathol_index_array  = np.arange(len(pathol_name_class_combo))
pathol_folds        = [(train_validate_index, test_index) for train_validate_index, test_index in pathol_spliter.split(pathol_index_array)]

file_results        = []
snippet_results     = []
total_file_con_mat  = np.array([[0,0],[0,0]])
total_snip_con_mat  = np.array([[0,0],[0,0]])

data_file_name       = input_name + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" +                  "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
aug_dict_file_name   = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" +                  "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
unaug_dict_file_name = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)

temp_file_1          = open(dataset_path + '/' + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path + '/' + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path + '/' + unaug_dict_file_name + '.pickle', 'rb')
data                 = pickle.load(temp_file_1)
aug_dict             = pickle.load(temp_file_2)
unaug_dict           = pickle.load(temp_file_3)




encoder     = autoencoderRandomWeights.main(input_vector_length, arch_bundle)

for fold_index in range(num_folds):
    print("---> Now Fold ", fold_index + 1,  " ----------------------------")

    normal_train_validate_combo = normal_name_class_combo[normal_folds[fold_index][0]]
    normal_test_combo           = normal_name_class_combo[normal_folds[fold_index][1]]
    normal_train_validate_combo = normal_train_validate_combo.tolist()
    normal_test_combo           = normal_test_combo.tolist()
    
    [normal_train_combo, normal_validate_combo, _] = dataSplit.main(normal_train_validate_combo, train_percent, 100 - train_percent, 0)


    pathol_train_validate_combo = pathol_name_class_combo[pathol_folds[fold_index][0]]
    pathol_test_combo           = pathol_name_class_combo[pathol_folds[fold_index][1]]
    pathol_train_validate_combo = pathol_train_validate_combo.tolist()
    pathol_test_combo           = pathol_test_combo.tolist()
    
    [pathol_train_combo, pathol_validate_combo, _] = dataSplit.main(pathol_train_validate_combo, train_percent, 100 - train_percent, 0)
    

    train_combo    = normal_train_combo    + pathol_train_combo
    validate_combo = normal_validate_combo + pathol_validate_combo
    test_combo     = normal_test_combo     + pathol_test_combo

    # =============================================================================
    train_package     = loadMelSpectrogram(train_combo,    classes, dsp_package, dataset_path, data, True,  aug_dict)   
    validate_package  = loadMelSpectrogram(validate_combo, classes, dsp_package, dataset_path, data, False, unaug_dict)   
    test_package      = loadMelSpectrogram(test_combo,     classes, dsp_package, dataset_path, data, False, unaug_dict)

    train_data,    _,  _, train_label3,    train_dist,    _                       = train_package
    validate_data, _,  _, validate_label3, validate_dist, validate_augment_amount = validate_package
    test_data,     _,  _, test_label3,     test_dist,     test_augment_amount     = test_package
    print(train_dist)
    print(validate_dist)
    print(test_dist)
    
    encoder     = autoencoderRandomWeights.main(input_vector_length, arch_bundle)

    train_data_encoded     = encoder.predict(train_data)
    validate_data_encoded  = encoder.predict(validate_data)
    test_data_encoded      = encoder.predict(test_data)

    fold_result_package  = mySVM.method1(train_data_encoded,    train_label3, 
                                         validate_data_encoded, validate_label3, 
                                         test_data_encoded,     test_label3,
                                         test_combo,            test_augment_amount)
    

    file_acc, file_con_mat, snippet_acc, snippet_con_mat = fold_result_package

    total_file_con_mat = total_file_con_mat + file_con_mat
    total_snip_con_mat = total_snip_con_mat + snippet_con_mat
    print(file_acc)
    print(snippet_acc)
    print(file_con_mat)
    print(snippet_con_mat)
    file_results.append(file_acc)
    snippet_results.append(snippet_acc)

    cur_possible_result = ((num_normal - total_file_con_mat[0][1]) / num_normal + (num_pathol - total_file_con_mat[1][0]) / num_pathol) / 2
    print('The best results we can get is:', cur_possible_result)


print('--------------------------------')
print('file results')
print(file_results)
print(sum(file_results) / len(file_results))
print('--------------------------------')
print('snippet results')
print(snippet_results)
print(sum(snippet_results) / len(snippet_results))
print('--------------------------------')
print('final file results')
print(total_file_con_mat)
acc = 0;
for i in range(len(total_file_con_mat[0])):
    acc = acc + total_file_con_mat[i][i] / sum(total_file_con_mat[i])
print(acc / 2)
print('--------------------------------')
print('final snippet results')
print(total_snip_con_mat)
acc = 0;
for i in range(len(total_snip_con_mat[0])):
    acc = acc + total_snip_con_mat[i][i] / sum(total_snip_con_mat[i])
print(acc / 2)    