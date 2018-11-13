from   sklearn.model_selection import KFold
import numpy                   as     np
import math
import getCombination
import mySVM
import dataSplit
import os
import pickle
from loadMFCCs import loadMFCCs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# Dataset Initialization
classes              = ['Normal','Pathol']
dataset_path_main    = "/home/hguan/7100-Master-Project/Dataset-KayPentax"
dataset_path_support = "/home/hguan/7100-Master-Project/Dataset-Spanish"
train_percent        = 90
num_folds            = 5
input_name           = "MFCCs"
slash                = '/'


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
name_class_combo    = getCombination.main(dataset_path_main, classes)
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
temp_file_1          = open(dataset_path_main + slash + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path_main + slash + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path_main + slash + unaug_dict_file_name + '.pickle', 'rb')
MFCCs_data_1         = pickle.load(temp_file_1)
aug_dict_1           = pickle.load(temp_file_2)
unaug_dict_1         = pickle.load(temp_file_3)

temp_file_1          = open(dataset_path_support + '/' + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path_support + '/' + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path_support + '/' + unaug_dict_file_name + '.pickle', 'rb')
MFCCs_data_2         = pickle.load(temp_file_1)
aug_dict_2           = pickle.load(temp_file_2)
unaug_dict_2         = pickle.load(temp_file_3)


# =============================================================================
name_class_combo_support = getCombination.main(dataset_path_support, classes)
train_combo_support      = np.asarray(name_class_combo_support)
train_package_support    = loadMFCCs(train_combo_support,  classes, dsp_package, dataset_path_support, MFCCs_data_2, True,  aug_dict_2)   


# =============================================================================
# n-folds Cross Validation Initialization
num_folds               = 5
kf_spliter              = KFold(n_splits = num_folds, shuffle = True)
name_class_combo        = getCombination.main(dataset_path_main, classes)
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

    # load all the snippet's spectrogram's (already saved before)
    train_package     = loadMFCCs(train_combo,    classes, dsp_package, dataset_path_main, MFCCs_data_1, True,  aug_dict_1)   
    validate_package  = loadMFCCs(validate_combo, classes, dsp_package, dataset_path_main, MFCCs_data_1, False, unaug_dict_1)   
    test_package      = loadMFCCs(test_combo,     classes, dsp_package, dataset_path_main, MFCCs_data_1, False, unaug_dict_1)
    
    train_data_main,    _,    _, train_label_3_main,     _,   _                         = train_package
    validate_data, _,    _, validate_label_3,  validate_dist, validate_augment_amount,  = validate_package
    test_data,     _,    _, test_label_3,      test_dist,     test_augment_amount,      = test_package
    train_data_support, _, _, train_label_3_support, _, _ = train_package_support
    # ==============================================================================
    train_data    = np.zeros((len(train_data_main) + len(train_data_support), 40))
    train_data[0 : len(train_data_main)] = train_data_main
    train_data[len(train_data_main) : ]  = train_data_support
    train_label_3 = np.concatenate((train_label_3_main,   train_label_3_support), axis = 0)


    print(train_data.shape)
    print(validate_data.shape)
    print(test_data.shape)
    #print(validate_augment_amount, test_augment_amount)
    train_data_normalized    = np.zeros((train_data.shape))
    validate_data_normalized = np.zeros((validate_data.shape))
    test_data_normalized     = np.zeros((test_data.shape))

    for i in range(input_vector_length):      
        mean                    = np.mean(train_data[:, i].flatten().tolist() + validate_data[:, i].flatten().tolist())
        std                     = np.std(train_data[:, i].flatten().tolist()  + validate_data[:, i].flatten().tolist())
        train_data_normalized[:, i]        = (train_data[:, i]    - mean) / std
        validate_data_normalized[:, i]     = (validate_data[:, i] - mean) / std
        test_data_normalized[:, i]         = (test_data[:, i]     - mean) / std

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