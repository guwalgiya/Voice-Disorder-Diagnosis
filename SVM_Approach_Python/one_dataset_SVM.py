from   sklearn.model_selection import KFold
import numpy                   as     np
import math
import getCombination
import mySVM
import dataSplit
import os
import pickle
from loadMFCCs import loadMFCCs

# =============================================================================
# Dataset Initialization
classes        = ['Normal','Pathol']
dataset_path   = "/home/hguan/7100-Master-Project/Dataset-Spanish"
train_percent      = 90
num_folds          = 5
input_name         = "MFCCs"
slash          = '/'


# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500  #in milliseconds
snippet_hop         = 100 #in ms
fft_length          = 512
fft_hop             = 128
input_vector_length = 40
mel_length          = 128
dsp_package         = [fs, snippet_length, snippet_hop, fft_length, fft_hop, input_vector_length]


# =============================================================================
# Get
name_class_combo    = getCombination.main(dataset_path, classes)
name_class_combo    = np.asarray(name_class_combo)
kf_spliter          = KFold(n_splits = num_folds, shuffle = True)
index_array         = np.arange(len(name_class_combo))
file_results        = []
snippet_results     = []
file_con_mat        = np.array([[0,0],[0,0]])
snippet_con_mat     = np.array([[0,0],[0,0]])

# =============================================================================
# Loading Data
data_file_name       = input_name + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop)
aug_dict_file_name   = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" 
unaug_dict_file_name = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" 
temp_file_1          = open(dataset_path + slash + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path + slash + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path + slash + unaug_dict_file_name + '.pickle', 'rb')
MFCCs_data           = pickle.load(temp_file_1)
aug_dict             = pickle.load(temp_file_2)
unaug_dict           = pickle.load(temp_file_3)

# =============================================================================
# n-folds Cross Validation Initialization
num_folds               = 5
kf_spliter              = KFold(n_splits = num_folds, shuffle = True)
name_class_combo        = getCombination.main(dataset_path, classes)
name_class_combo        = np.asarray(name_class_combo)
normal_name_class_combo = [x for x in name_class_combo if (x[1] == "Normal")]
pathol_name_class_combo = [x for x in name_class_combo if (x[1] == "Pathol")]
normal_index_array      = np.arange(len(normal_name_class_combo))
pathol_index_array      = np.arange(len(normal_name_class_combo), len(name_class_combo))
normal_split            = kf_spliter.split(normal_index_array)
pathol_split            = kf_spliter.split(pathol_index_array)


# =============================================================================
# Creat N-folds
normal_split_index = []
for train_validate_index, test_index in normal_split:
    normal_split_index.append([normal_index_array[train_validate_index], normal_index_array[test_index]])

pathol_split_index = [] # be very careful here
for train_validate_index, test_index in pathol_split:
    pathol_split_index.append([pathol_index_array[train_validate_index], pathol_index_array[test_index]])


# ==============================================================================
# start to do outer Leave-one-out Cross cross validation
for fold_num in range(num_folds):
    print("---> Now Fold ", fold_num + 1,  " ----------------------------")


    # ==============================================================================
    normal_train_validate_combo = name_class_combo[normal_split_index[fold_num][0]].tolist() 
    pathol_train_validate_combo = name_class_combo[pathol_split_index[fold_num][0]].tolist()  
    normal_test_combo           = name_class_combo[normal_split_index[fold_num][1]].tolist()  
    pathol_test_combo           = name_class_combo[pathol_split_index[fold_num][1]].tolist()  


    # ==============================================================================
    [normal_train_combo, normal_validate_combo, _] = dataSplit.main(normal_train_validate_combo, train_percent, 100 - train_percent, 0) 
    [pathol_train_combo, pathol_validate_combo, _] = dataSplit.main(pathol_train_validate_combo, train_percent, 100 - train_percent, 0)
    

    # ==============================================================================
    train_combo    = normal_train_combo    + pathol_train_combo
    validate_combo = normal_validate_combo + pathol_validate_combo
    test_combo     = normal_test_combo     + pathol_test_combo
    

    # =============================================================================================
    # load all the snippet's spectrogram's (already saved before)
    train_package     = loadMFCCs(train_combo,    classes, dsp_package, MFCCs_data, True,  aug_dict)   
    validate_package  = loadMFCCs(validate_combo, classes, dsp_package, MFCCs_data, False, unaug_dict)   
    test_package      = loadMFCCs(test_combo,     classes, dsp_package, MFCCs_data, False, unaug_dict)
    

    # =============================================================================================
    train_data,    _,    _, train_label_3,     train_dist,    _                         = train_package
    validate_data, _,    _, validate_label_3,  validate_dist, validate_augment_amount,  = validate_package
    test_data,     _,    _, test_label_3,      test_dist,     test_augment_amount,      = test_package
    

    # =============================================================================================
    print(train_dist)
    print(validate_dist)
    print(test_dist)


    # =============================================================================================
    train_data_normalized    = np.zeros((train_data.shape))
    validate_data_normalized = np.zeros((validate_data.shape))
    test_data_normalized     = np.zeros((test_data.shape))
    for i in range(input_vector_length):      
        mean                    = np.mean(train_data[:, i].flatten().tolist() + validate_data[:, i].flatten().tolist())
        std                     = np.std(train_data[:, i].flatten().tolist()  + validate_data[:, i].flatten().tolist())
        max_0                   = np.max(train_data[:, i].flatten().tolist()  + validate_data[:, i].flatten().tolist())
        min_0                   = np.min(train_data[:, i].flatten().tolist()  + validate_data[:, i].flatten().tolist())
        #print("feature", i + 1)
        #print(mean, std, max_0, min_0)
        train_data_normalized[:, i]        = (train_data[:, i]    - min_0)  / (max_0 - min_0)
        validate_data_normalized[:, i]     = (validate_data[:, i] - min_0)  / (max_0 - min_0)
        test_data_normalized[:, i]         = (test_data[:, i]     - min_0)  / (max_0 - min_0)
        np.clip(test_data_normalized[:, i], 0, 1)


    # =============================================================================================
    cur_result_package = mySVM.method1(train_data_normalized,    train_label_3, 
                                       validate_data_normalized, validate_label_3, 
                                       test_data_normalized,     test_label_3,
                                       test_combo,               test_augment_amount)
    cur_file_acc, cur_file_con_mat, cur_snippet_acc, cur_snippet_con_mat = cur_result_package
    print(cur_file_acc, cur_snippet_acc)
    print(cur_file_con_mat)
    print(cur_snippet_con_mat)
    file_results.append(cur_file_acc)
    snippet_results.append(cur_snippet_acc)
    file_con_mat    = file_con_mat    + cur_file_con_mat
    snippet_con_mat = snippet_con_mat + cur_snippet_con_mat

    # ==============================================================================
    # Predict the future
    cur_possible_result_Normal = (len(normal_name_class_combo) - file_con_mat[0][1]) / len(normal_name_class_combo) / 2 
    cur_possible_result_Pathol = (len(pathol_name_class_combo) - file_con_mat[1][0]) / len(pathol_name_class_combo) / 2
                       
    print('The best results we can get is:', cur_possible_result_Normal + cur_possible_result_Pathol)

# ==============================================================================
# show final results
print('--------------------------------')
print('file results')
print(sum(file_results) / len(file_results))
print('--------------------------------')
print('snippet results')
print(sum(snippet_results) / len(snippet_results))
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