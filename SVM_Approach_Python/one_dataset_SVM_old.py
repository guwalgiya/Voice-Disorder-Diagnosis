from   sklearn.model_selection import KFold
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
classes = ['Normal','Pathol']
dataset_main_path = "/home/hguan/7100-Master-Project/Dataset-KeyPentax"
train_percent      = 90
num_folds          = 5
# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500  #in milliseconds
snippet_hop         = 100 #in ms
block_size          = 512
hop_size            = 128
package             = [snippet_length, snippet_hop, block_size, hop_size]
input_vector_length = 40
# =============================================================================
# Get
name_class_combo    = getCombination.main(dataset_main_path, classes)
name_class_combo    = np.asarray(name_class_combo)
kf_spliter          = KFold(n_splits = num_folds, shuffle = True)
index_array         = np.arange(len(name_class_combo))
file_results        = []
snippet_results     = []
total_file_con_mat  = np.array([[0,0],[0,0]])
total_snip_con_mat  = np.array([[0,0],[0,0]])

for train_val_index, test_index in kf_spliter.split(index_array):
    print('New Fold---------------------------------')
    train_val_combo = name_class_combo[train_val_index]
    test_combo      = name_class_combo[test_index]
    train_val_combo = train_val_combo.tolist()  
    test_combo      = test_combo.tolist()

    [train_combo, validate_combo, _] = dataSplit.main(train_val_combo, train_percent, 100 - train_percent, 0)

    # load all the snippet's spectrogram's (already saved before)
    train_package     = loadMFCCs.main(train_combo,    'training',   classes, package, fs, dataset_main_path, input_vector_length)   
    validate_package  = loadMFCCs.main(validate_combo, 'validating', classes, package, fs, dataset_main_path, input_vector_length)   
    test_package      = loadMFCCs.main(test_combo,     'testing',    classes, package, fs, dataset_main_path, input_vector_length)
    
    train_data,    _,    train_label2,    train_dist,    _                       = train_package
    validate_data, _,    validate_label2, validate_dist, validate_augment_amount = validate_package
    test_data,     _,    test_label2,     test_dist,     test_augment_amount     = test_package
    
    print(train_dist, validate_dist, test_dist)
    #print(validate_augment_amount, test_augment_amount)
    train_data_normalized    = np.zeros((train_data.shape))
    validate_data_normalized = np.zeros((validate_data.shape))
    test_data_normalized     = np.zeros((test_data.shape))

    for i in range(input_vector_length):
        standard = max(max(abs(train_data[:, i])), max(abs(validate_data[:, i])))
        train_data_normalized[:, i]    = train_data[:,i]      / float(standard)
        validate_data_normalized[:, i] = validate_data[:, i]  / float(standard)
        test_data_normalized[:, i]     = test_data[:,i]       / float(standard)



    file_acc, snippet_acc, file_con_mat,  snippet_con_mat = mySVM.method1(train_data_normalized,    train_label2, 
                                                                          validate_data_normalized, validate_label2, 
                                                                          test_data_normalized,     test_label2,
                                                                          test_combo,               test_augment_amount,
                                                                          validate_combo,           validate_augment_amount)
    print(file_acc, snippet_acc)
    print(file_con_mat)
    print(snippet_con_mat)
    total_file_con_mat = total_file_con_mat + file_con_mat
    total_snip_con_mat = total_snip_con_mat + snippet_con_mat
    file_results.append(file_acc)
    snippet_results.append(snippet_acc)

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