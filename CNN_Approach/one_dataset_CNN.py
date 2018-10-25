# =============================================================================
# Import Packages
import tensorflow              as     tf
import numpy                   as     np 
import os
import CNN
import pickle
import dataSplit
import matplotlib
import mySVM
matplotlib.use('Agg')
import getCombination
from   matplotlib              import pyplot             as plt
from   keras                   import backend            as K
from   loadMelSpectrogram      import loadMelSpectrogram
from   resultsAnalysisCNN      import resultsAnalysis
from   keras.utils             import plot_model
from   keras.models            import load_model, Model
from   sklearn.model_selection import KFold
from   math                    import ceil

# =============================================================================
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
sess        = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))


# =============================================================================
# Environment Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dataset_path                       = "/home/hguan/7100-Master-Project/Dataset-Spanish"
slash                              = "/"
#dataset_path                      = "C:\\Master Degree\\7100 - Master Project\\Dataset - KayPentax"
#slash                             = "\\"
val_loss_plot_name                 = "Val_Loss_Plot_"
best_model_name                    = "best_model_this_fold.hdf5"


# =============================================================================
# Dataset Initialization
classes     = ['Normal','Pathol']
input_name  = "MelSpectrogram"


# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500   #in milliseconds
snippet_hop         = 100   #in ms
fft_length          = 512
fft_hop             = 128
mel_length          = 128
num_MFCCs           = 20
num_rows            = 128
dsp_package         = [fs, snippet_length, snippet_hop, fft_length, fft_hop]
#input_vector_length = mel_length * math.ceil(snippet_length / 1000 * fs / fft_hop)
input_vector_length = num_rows * ceil(snippet_length / 1000 * fs / fft_hop)
input_name          = "MelSpectrogram"

# =============================================================================
# Deep Learning Initialization
fold_num      = 1
train_percent = 90
epoch_limit   = 100000
batch_size    = 1024
num_channel   = 4
input_shape   = (int(num_rows / num_channel), ceil(snippet_length / 1000 * fs / fft_hop), num_channel)
monitor       = "val_loss"


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


# =============================================================================
# Loading Data
data_file_name       = input_name + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
aug_dict_file_name   = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" 
unaug_dict_file_name = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" 
temp_file_1          = open(dataset_path + slash + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path + slash + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path + slash + unaug_dict_file_name + '.pickle', 'rb')
data                 = pickle.load(temp_file_1)
aug_dict             = pickle.load(temp_file_2)
unaug_dict           = pickle.load(temp_file_3)


# =============================================================================
# Results Initialization
file_results     = []
snippet_results  = []
file_con_mat     = np.array([[0,0],[0,0]])
snippet_con_mat  = np.array([[0,0],[0,0]])

file_results0     = []
snippet_results0  = []
file_con_mat0     = np.array([[0,0],[0,0]])
snippet_con_mat0  = np.array([[0,0],[0,0]])


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


    # ==============================================================================
    train_package     = loadMelSpectrogram(train_combo,    classes, dsp_package, num_rows, "melSpec", data, False, unaug_dict)
    validate_package  = loadMelSpectrogram(validate_combo, classes, dsp_package, num_rows, "melSpec", data, False, unaug_dict)   
    test_package      = loadMelSpectrogram(test_combo,     classes, dsp_package, num_rows, "melSpec", data, False, unaug_dict)
    

    # ==============================================================================
    train_data,    _,            train_label_2,    train_label_3,    train_dist,    _                   = train_package
    validate_data, _,            validate_label_2, validate_label_3, validate_dist, _                   = validate_package
    test_data,     test_label_1, test_label_2,     test_label_3,     test_dist,     test_augment_amount = test_package
    

    # ==============================================================================
    train_data    = train_data.reshape(train_data.shape[0],       num_channel, int(train_data.shape[1]    / num_channel), train_data.shape[2])   
    validate_data = validate_data.reshape(validate_data.shape[0], num_channel, int(validate_data.shape[1] / num_channel), validate_data.shape[2]) 
    test_data     = test_data.reshape(test_data.shape[0],         num_channel, int(test_data.shape[1]     / num_channel), test_data.shape[2]) 
    

    # ==============================================================================
    train_data    = np.moveaxis(train_data,    1, -1)
    validate_data = np.moveaxis(validate_data, 1, -1)
    test_data     = np.moveaxis(test_data,     1, -1)


    # ==============================================================================
    _, history = CNN.main(train_data, train_label_2, train_label_3, validate_data, validate_label_2, validate_label_3, epoch_limit, batch_size, input_shape, monitor)
    best_CNN   = load_model(best_model_name)
    
    
    # ==============================================================================
    # save the plot of validation loss
    plt.plot(history.history[monitor])
    plt.savefig(val_loss_plot_name + str(fold_num + 1) + ".png")
    plt.clf()

    # ==============================================================================
    cur_result_package = resultsAnalysis(best_CNN, test_combo, test_data, test_label_1, test_augment_amount, classes)
    cur_file_acc, cur_snippet_acc, cur_file_con_mat, cur_snippet_con_mat = cur_result_package
    print('----------------------------')
    print(cur_file_acc)
    print(cur_snippet_acc)
    print(cur_file_con_mat)
    print(cur_snippet_con_mat)
    print("Now start to use SVM")
    
    file_results0.append(cur_file_acc)
    snippet_results0.append(cur_snippet_acc)
    file_con_mat0    = file_con_mat0    + cur_file_con_mat
    snippet_con_mat0 = snippet_con_mat0 + cur_snippet_con_mat

    # =============================================================================
    best_dim         = 0
    best_file_acc    = 0
    possible_encoder = []
    best_result_pack = None
    print("Feature Extractor is trained")
    for dim in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        index     = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].index(dim)
        extractor = Model(inputs  = best_CNN.inputs, outputs = best_CNN.layers[-1 - index].output)

        
        # =============================================================================
        train_data_CNNed    = extractor.predict(train_data)
        validate_data_CNNed = extractor.predict(validate_data)
        test_data_CNNed     = extractor.predict(test_data)
        
        # =============================================================================
        fold_result_package  = mySVM.method1(train_data_CNNed,    train_label_3, 
                                             validate_data_CNNed, validate_label_3, 
                                             test_data_CNNed,     test_label_3,
                                             test_combo,          test_augment_amount)
    
        cur_file_acc, cur_file_con_mat, cur_snippet_acc, cur_snippet_con_mat = fold_result_package
        print(dim, cur_file_acc)
        if cur_file_acc > best_file_acc:
            best_file_acc    = cur_file_acc
            best_result_pack = fold_result_package
            best_dim         = dim

   
    # =============================================================================
    cur_file_acc, cur_file_con_mat, cur_snippet_acc, cur_snippet_con_mat = best_result_pack
    print('----------------------------')
    print(cur_file_acc)
    print(cur_snippet_acc)
    print(cur_file_con_mat)
    print(cur_snippet_con_mat)

    # =============================================================================
    file_results.append(cur_file_acc)
    snippet_results.append(cur_snippet_acc)
    file_con_mat    = file_con_mat    + cur_file_con_mat
    snippet_con_mat = snippet_con_mat + cur_snippet_con_mat
    fold_num        = fold_num        + 1


    # ==============================================================================
    K.clear_session()
    tf.reset_default_graph()
    os.remove("best_model_this_fold.hdf5")
    print("Memory Cleared, Saved Model Removed")


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


# ==============================================================================
# show final results
print('--------------------------------')
print('file results')
print(sum(file_results0) / len(file_results0))
print('--------------------------------')
print('snippet results')
print(sum(snippet_results0) / len(snippet_results0))
print('--------------------------------')
print('final file results')
print(file_con_mat0)
acc = 0;
for i in range(len(file_con_mat0[0])):
    acc = acc + file_con_mat0[i][i] / sum(file_con_mat0[i])
print(acc / 2)
print('--------------------------------')
print('final snippet results')
print(snippet_con_mat0)
acc = 0;
for i in range(len(snippet_con_mat0[0])):
    acc = acc + snippet_con_mat0[i][i] / sum(snippet_con_mat0[i])
print(acc / 2)