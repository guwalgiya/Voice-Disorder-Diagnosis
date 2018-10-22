# =============================================================================
# Import Packages
from   keras                   import backend            as K
from   loadMelSpectrogram      import loadMelSpectrogram
from   sklearn.model_selection import KFold
from   math                    import ceil
import tensorflow              as     tf
import numpy                   as     np 
import os
import CNN
import pickle
import dataSplit
import getCombination
import resultsAnalysis

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =============================================================================
# Dataset Initialization
classes       = ['Normal','Pathol']
dataset_path  = "/home/hguan/7100-Master-Project/Dataset-KayPentax"
input_name    = "MelSpectrogram"
result_name   = "one_dataset_CNN_results"


# =============================================================================
# Dsp Initialization
fs             = 16000
snippet_length = 500   #in milliseconds
snippet_hop    = 100   #in ms
fft_length     = 512
fft_hop        = 128
mel_length     = 128
dsp_package    = [fs, snippet_length, snippet_hop, fft_length, fft_hop, mel_length]


# =============================================================================
# Deep Learning Initialization
file_num      = 1
train_percent = 90
epoch_limit   = 10000
num_channel   = 4
input_shape   = (num_channel, int(mel_length / num_channel), ceil(snippet_length / 1000 * fs / fft_hop))


# =============================================================================
# Leave-one-out Cross Validation Initialization
num_folds        = 655 + 53
name_class_combo = getCombination.main(dataset_path, classes)
name_class_combo = np.asarray(name_class_combo)
kf_spliter       = KFold(n_splits = num_folds, shuffle = True)
index_array      = np.arange(len(name_class_combo))


# =============================================================================
# Loading Data
data_file_name       = input_name + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
aug_dict_file_name   = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" 
unaug_dict_file_name = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" 
temp_file_1          = open(dataset_path + '/' + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path + '/' + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path + '/' + unaug_dict_file_name + '.pickle', 'rb')
data                 = pickle.load(temp_file_1)
aug_dict             = pickle.load(temp_file_2)
unaug_dict           = pickle.load(temp_file_3)


# =============================================================================
# Results Initialization
file_results     = []
snippet_results  = []
file_con_mat     = np.array([[0,0],[0,0]])
snippet_con_mat  = np.array([[0,0],[0,0]])


# ==============================================================================
try:
    temp_file_0         = open(dataset_path + "/" + result_name + ".pickle", "rb" )
    results             = pickle.load(temp_file_0)
    file_results, snippet_results, file_con_mat, snippet_con_mat = results["Results"]
except:
    results             = {}
    results["Results"]  = [file_results, snippet_results, file_con_mat, snippet_con_mat]


# ==============================================================================
# start to do outer Leave-one-out Cross cross validation
for train_validate_index, test_index in kf_spliter.split(index_array):
    print("---> Now Fold ", file_num,  " ----------------------------")


    # ==============================================================================
    test_combo   = name_class_combo[test_index]    
    test_combo   = test_combo.tolist()
    test_name    = test_combo[0][0]
    test_class   = test_combo[0][1]

    
    # ==============================================================================
    if (test_name in results.keys()) and (results[test_name][0] == 1):
        print("We already did ", test_name)
        file_num = file_num + 1


    # ==============================================================================
    else:
        results[test_name] = []


        # ==============================================================================
        train_validate_combo        = name_class_combo[train_validate_index].tolist() 
        normal_train_validate_combo = [x for x in train_validate_combo if (x[1] == "Normal")]
        pathol_train_validate_combo = [x for x in train_validate_combo if (x[1] == "Pathol")]


        # ==============================================================================
        [normal_train_combo, normal_validate_combo, _] = dataSplit.main(normal_train_validate_combo, train_percent, 100 - train_percent, 0) 
        [pathol_train_combo, pathol_validate_combo, _] = dataSplit.main(pathol_train_validate_combo, train_percent, 100 - train_percent, 0)
        
        
        # ==============================================================================
        train_combo    = normal_train_combo    + pathol_train_combo
        validate_combo = normal_validate_combo + pathol_validate_combo


        # ==============================================================================
        train_package     = loadMelSpectrogram(train_combo,    classes, dsp_package, dataset_path, data, True,  aug_dict)
        validate_package  = loadMelSpectrogram(validate_combo, classes, dsp_package, dataset_path, data, False, unaug_dict)   
        test_package      = loadMelSpectrogram(test_combo,     classes, dsp_package, dataset_path, data, False, unaug_dict)


        # ==============================================================================
        train_data,    train_label_1,    _, _, train_dist,    _ = train_package
        validate_data, validate_label_1, _, _, validate_dist, _ = validate_package
        test_data,     _,                _, _, test_dist,     _ = test_package
        

        # ==============================================================================
        train_data    = train_data.reshape(train_data.shape[0],       num_channel, int(train_data.shape[1]    / num_channel), train_data.shape[2])   
        validate_data = validate_data.reshape(validate_data.shape[0], num_channel, int(validate_data.shape[1] / num_channel), validate_data.shape[2]) 
        test_data     = test_data.reshape(test_data.shape[0],         num_channel, int(test_data.shape[1]     / num_channel), test_data.shape[2]) 
        
     
        # ==============================================================================
        trained_CNN   = CNN.main(train_data, train_label_1, validate_data, validate_label_1, epoch_limit, input_shape)
         

        # ==============================================================================
        results[test_name] = resultsAnalysis.main(trained_CNN, test_combo, test_data)
        cur_file_acc, cur_snippet_acc, cur_file_con_mat, cur_snippet_con_mat = results[test_name]
        print(cur_file_acc)
        print(cur_snippet_acc)
        print(cur_file_con_mat)
        print(cur_snippet_con_mat)


        # ==============================================================================
        file_results.append(cur_file_acc)
        snippet_results.append(cur_snippet_acc)
        file_con_mat    = sum(results[file_name][2] for file_name in results.keys() if file_name != "Results")
        snippet_con_mat = sum(results[file_name][3] for file_name in results.keys() if file_name != "Results")
        file_num        = file_num        + 1
    

        # ==============================================================================
        results["Results"] = [file_results,  snippet_results,  file_con_mat,      snippet_con_mat]

        with open(dataset_path + "/" + result_name + ".pickle", "wb") as temp_file_4:
                pickle.dump(results, temp_file_4, protocol = pickle.HIGHEST_PROTOCOL)


        # ==============================================================================
        K.clear_session()
        tf.reset_default_graph()


    # ==============================================================================
    # Predict the future
    cur_possible_result = ((53 - file_con_mat[0][1]) / 53 + (655 - file_con_mat[1][0]) / 655) / 2
    print('The best results we can get is:', cur_possible_result)


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