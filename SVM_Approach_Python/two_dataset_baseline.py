# =============================================================================
# Import Packages
from   loadMFCCs      import loadMFCCs
from   math           import ceil
import numpy          as     np 
import getCombination
import os
import pickle
import dataSplit
import mySVM


# =============================================================================
# Dataset Initialization
classes            = ["Normal", "Pathol"]
dataset_path_train = "/home/hguan/7100-Master-Project/Dataset-Spanish"
dataset_path_test  = "/home/hguan/7100-Master-Project/Dataset-KayPentax"
train_percent      = 90
validate_percent   = 10
test_percent       = 0


# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500   #in milliseconds
snippet_hop         = 100   #in ms
fft_length          = 512
fft_hop             = 128
mel_length          = 128
num_MFCCs           = 20 * 2
num_rows            = num_MFCCs
dsp_package         = [fs, snippet_length, snippet_hop, fft_length, fft_hop, num_rows]
input_name          = "MFCCs"


# =============================================================================
data_file_name       = input_name + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop)
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
validate_combo = normal_validate_combo    + pathol_validate_combo


# =============================================================================
test_combo          = getCombination.main(dataset_path_test,  classes)
[_, _, test_combo]  = dataSplit.main(test_combo, 0, 0, 100)
test_combo          = test_combo

# =============================================================================
train_package     = loadMFCCs(train_combo,    classes, dsp_package, data_1, True,  aug_dict_1)   
validate_package  = loadMFCCs(validate_combo, classes, dsp_package, data_1, False, unaug_dict_1)   
test_package      = loadMFCCs(test_combo,     classes, dsp_package, data_2, False, unaug_dict_2)


# =============================================================================
train_data,    train_label_1,    _, _, train_dist,    _                   = train_package
validate_data, validate_label_1, _, _, validate_dist, _                   = validate_package
test_data,     test_label_1,     _, _, test_dist,     test_augment_amount = test_package
print(train_dist)

print(train_label_1.count(0))
print(train_label_1.count(1))
print(test_dist)
print(test_label_1.count(0))
print(test_label_1.count(1))
# =============================================================================
train_data_normalized    = np.zeros((train_data.shape))
validate_data_normalized = np.zeros((validate_data.shape))
test_data_normalized     = np.zeros((test_data.shape))
for i in range(num_rows):      
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
#   print(np.mean(test_data[:, i]), np.std(test_data[:, i]), np.max(test_data[:, i]), np.min(test_data[:, i]))
# print("Waaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
# for i in range(50):
#     print(i, '-------------------------------')
#     print(test_data[i, :]) 
#     print(test_data_normalized[i, :])
# # =============================================================================
result_package  = mySVM.method1(train_data_normalized,    train_label_1, 
                                validate_data_normalized, validate_label_1, 
                                test_data_normalized,     test_label_1,
                                test_combo,               test_augment_amount)




# =============================================================================
file_acc, file_con_mat, snippet_acc, snippet_con_mat = result_package


# =============================================================================
# Report Final Results
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