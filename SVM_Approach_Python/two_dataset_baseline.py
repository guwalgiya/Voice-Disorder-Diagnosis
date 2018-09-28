import numpy                   as     np
import math
import getCombination
import mySVM
import dataSplit
import os
import loadMFCCs
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
bin_size            = 500
block_size          = 512
hop_size            = 128
package             = [snippet_length, snippet_hop, block_size, hop_size]
input_vector_length = 40
# ===========================
# Get
# Get
name_class_combo_train = getCombination.main(dataset_main_path_train, classes)
name_class_combo_test  = getCombination.main(dataset_main_path_test,  classes)
# =============================================================================
[train_combo, validate_combo, test_combo]  = dataSplit.main(name_class_combo_train, train_percent, validate_percent, test_percent)
[_,  _,              test_combo]           = dataSplit.main(name_class_combo_test,  0,             0,                100)
# =============================================================================
train_package     = loadMFCCs.main(train_combo,    'training',   classes, package, fs, dataset_main_path_train, input_vector_length)   
validate_package  = loadMFCCs.main(validate_combo, 'validating', classes, package, fs, dataset_main_path_train, input_vector_length)   
test_package      = loadMFCCs.main(test_combo,     'testing',    classes, package, fs, dataset_main_path_test,  input_vector_length)

train_data,    _,    train_label2,    train_dist,    _                       = train_package
validate_data, _,    validate_label2, validate_dist, validate_augment_amount = validate_package
test_data,     _,    test_label2,     test_dist,     test_augment_amount     = test_package
    
print(train_dist, validate_dist, test_dist)
train_data_normalized    = np.zeros((train_data.shape))
validate_data_normalized = np.zeros((validate_data.shape))
test_data_normalized     = np.zeros((test_data.shape))

for i in range(input_vector_length):
    standard = max(max(abs(train_data[:, i])), max(abs(validate_data[:, i])))
    train_data_normalized[:, i]    = train_data[:,i]      / float(standard)
    validate_data_normalized[:, i] = validate_data[:, i]  / float(standard)
    test_data_normalized[:, i]     = test_data[:,i]       / float(standard)



file_acc, snippet_acc, total_file_con_mat,  total_snippet_con_mat = mySVM.method1(train_data_normalized,    train_label2, 
                                                                      validate_data_normalized, validate_label2, 
                                                                      test_data_normalized,     test_label2,
                                                                      test_combo,               test_augment_amount,
                                                                      validate_combo,           validate_augment_amount)

print('--------------------------------')
print('file results')
print(file_acc)
print('--------------------------------')
print('snippet results')
print(snippet_acc)
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