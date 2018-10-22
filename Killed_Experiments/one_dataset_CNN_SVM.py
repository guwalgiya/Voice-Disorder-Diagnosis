from   sklearn.model_selection import KFold
from   keras.models            import Model
import numpy                   as     np
import math
import getCombination
import mySVM
import dataSplit
import loadMelSpectrogram
import os
import CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# Machine Learning Initialization
classes = ['Normal','Pathol']
dataset_main_path  = "/home/hguan/7100-Master-Project/Dataset-KeyPentax"
train_percent      = 90
num_folds          = 5
epoch_limit        = 10000

# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500   #in milliseconds
snippet_hop         = 100   #in ms
mel_length          = 128
block_size          = 512
hop_size            = 128
dsp_package         = [snippet_length, snippet_hop, block_size, hop_size, mel_length]
input_shape         = (mel_length, math.ceil(snippet_length / 1000 * fs / hop_size),1)
input_shape         = (20, 63, 1)
vector_length       = 20

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


for train_validate_index, test_index in kf_spliter.split(index_array):
    
    print('New Fold~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    train_validate_combo = name_class_combo[train_validate_index]
    train_validate_combo = train_validate_combo.tolist()  
    
    test_combo      = name_class_combo[test_index]
    test_combo      = test_combo.tolist()
    
    [train_combo, validate_combo, _] = dataSplit.main(train_validate_combo, train_percent, 100 - train_percent, 0)

    # load all the snippet's spectrogram's (already saved before)
    train_package     = loadMelSpectrogram.main(train_combo,    'training',    classes, dsp_package, fs, dataset_main_path)   
    validate_package  = loadMelSpectrogram.main(validate_combo, 'validating', classes, dsp_package, fs, dataset_main_path)   
    #train_validate_package = loadMelSpectrogram.main(train_validate_combo,    'trainng',    classes, dsp_package, fs, dataset_main_path)   
    test_package           = loadMelSpectrogram.main(test_combo,     'testing',    classes, dsp_package, fs, dataset_main_path)

    train_data,    train_label,    train_label2,    train_dist,    _                       = train_package
    validate_data, validate_label, validate_label2, validate_dist, validate_augment_amount = validate_package
    test_data,     test_label,     test_label2,     test_dist,     test_augment_amount     = test_package
    print(train_dist, validate_dist, test_dist)

    #train_validate_data,  train_validate_label, train_validate_label2,   _,    _,  = train_validate_package        

    #number_train    = round(len(train_validate_data) * train_percent / 100)
    #number_vlaidate = len(train_validate_data) - number_train
    
    # split_index     = np.random.permutation(len(train_validate_data))
    # train_index     = split_index[0 : number_train]
    # validate_index  = split_index[number_train : ]

    # train_data   = train_validate_data[train_index]
    # train_label  = list(train_validate_label[i]  for i in train_index)
    # train_label2 = list(train_validate_label2[i] for i in train_index)
    
    # validate_data   = train_validate_data[validate_index]
    # validate_label  = list(train_validate_label[i]  for i in validate_index)
    # validate_label2 = list(train_validate_label2[i] for i in validate_index)

    for i in range(vector_length):

        max_standard           = max(np.amax(train_data[:, i, :]), np.amax(validate_data[:, i, :]))
        min_standard           = min(np.amin(train_data[:, i, :]), np.amin(validate_data[:, i, :]))
        
        train_data[:, i, :]    = (train_data[:, i, :]    - min_standard) / (max_standard - min_standard)
        validate_data[:, i, :] = (validate_data[:, i, :] - min_standard) / (max_standard - min_standard)
        test_data[:, i, :]     = (test_data[:, i, :]     - min_standard) / (max_standard - min_standard)
        test_data[:, i, :]     = np.clip(test_data[:, i, :], 0, 1)
    


    train_data    = train_data.reshape(train_data.shape[0],       train_data.shape[1],    train_data.shape[2],    1)   
    validate_data = validate_data.reshape(validate_data.shape[0], validate_data.shape[1], validate_data.shape[2], 1) 
    test_data     = test_data.reshape(test_data.shape[0],         test_data.shape[1],     test_data.shape[2],     1) 
  

    cur_CNN       = CNN.main(train_data, train_label, validate_data, validate_label, epoch_limit, input_shape)


    feature_extractor = Model(inputs  = cur_CNN.input, outputs = cur_CNN.layers[-2].output)


    train_intermediate    = feature_extractor.predict(train_data)
    validate_intermediate = feature_extractor.predict(validate_data)
    test_intermediate     = feature_extractor.predict(test_data)

    print(test_intermediate.shape)
    result_package = mySVM.method1(train_intermediate,    train_label2, 
                                   validate_intermediate, validate_label2, 
                                   test_intermediate,     test_label2,
                                   test_combo,            test_augment_amount)


    file_acc, file_con_mat, snippet_acc, snippet_con_mat = result_package

    print('Fold Results')
    print(file_acc)
    print(snippet_acc)
    print(file_con_mat)
    print(snippet_con_mat)

    total_file_con_mat = total_file_con_mat + file_con_mat
    total_snip_con_mat = total_snip_con_mat + snippet_con_mat
    
    cur_possible_result = ((53 - total_file_con_mat[0][1]) / 53 + (655 - total_file_con_mat[1][0]) / 655) / 2
    

    file_results.append(file_acc)
    snippet_results.append(snippet_acc)
    
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
    