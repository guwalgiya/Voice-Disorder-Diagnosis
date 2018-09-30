# =============================================================================
# Import Packages
import matplotlib
matplotlib.use('Agg')
import getCombination
from   matplotlib                import pyplot             as plt
from   keras                     import backend            as K
from   sklearn.model_selection   import KFold
from   sklearn.feature_selection import SelectKBest, chi2
from   keras.models              import load_model
from   loadMelSpectrogram        import loadMelSpectrogram
from   loadMFCCs                 import loadMFCCs
from   keras.models              import Model
import numpy                     as     np
import tensorflow                as     tf
import math
import dataSplit
import autoencoder
import mySVM
import pickle
import os

# =============================================================================
# Dataset Initialization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
classes            = ["Normal", "Pathol"]
dataset_name       = "KayPentax"
dataset_path       = "/home/hguan/7100-Master-Project/Dataset-" + dataset_name
num_folds          = 5   
train_percent      = 90
best_model_name    = "best_model_this_fold.hdf5"
val_loss_plot_name = "Val_Loss_Plot_"
monitor            = "val_loss"


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
encoding_dimension = 128
encoder_layer      = 3
decoder_layer      = 3
epoch_limit        = 100000
batch_auto         = 1024
shuffle_choice     = True
loss_function      = 'mean_squared_error'
arch_bundle        = [encoder_layer, encoding_dimension, decoder_layer]
train_bundle_auto  = [epoch_limit, batch_auto, shuffle_choice, loss_function]


# =============================================================================
name_class_combo        = getCombination.main(dataset_path, classes)
normal_name_class_combo = [combo for combo in name_class_combo if combo[1] == "Normal"]
pathol_name_class_combo = [combo for combo in name_class_combo if combo[1] == "Pathol"]
normal_name_class_combo = np.asarray(normal_name_class_combo)
pathol_name_class_combo = np.asarray(pathol_name_class_combo)
num_normal              = len(normal_name_class_combo)
num_pathol              = len(pathol_name_class_combo)


# =============================================================================
normal_spliter      = KFold(n_splits = num_folds, shuffle = True)
normal_index_array  = np.arange(len(normal_name_class_combo))
normal_folds        = [(train_validate_index, test_index) for train_validate_index, test_index in normal_spliter.split(normal_index_array)]
pathol_spliter      = KFold(n_splits = num_folds, shuffle = True)
pathol_index_array  = np.arange(len(pathol_name_class_combo))
pathol_folds        = [(train_validate_index, test_index) for train_validate_index, test_index in pathol_spliter.split(pathol_index_array)]


# =============================================================================
file_results        = []
snippet_results     = []
total_file_con_mat  = np.array([[0,0],[0,0]])
total_snip_con_mat  = np.array([[0,0],[0,0]])


# =============================================================================
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
for fold_index in range(num_folds):
    print("---> Now Fold ", fold_index + 1,  " ----------------------------")
    

    # =============================================================================
    normal_train_validate_combo = normal_name_class_combo[normal_folds[fold_index][0]].tolist()
    normal_test_combo           = normal_name_class_combo[normal_folds[fold_index][1]].tolist()
    pathol_train_validate_combo = pathol_name_class_combo[pathol_folds[fold_index][0]].tolist()
    pathol_test_combo           = pathol_name_class_combo[pathol_folds[fold_index][1]].tolist()
    

    # =============================================================================
    [normal_train_combo, normal_validate_combo, _] = dataSplit.main(normal_train_validate_combo, train_percent, 100 - train_percent, 0)
    [pathol_train_combo, pathol_validate_combo, _] = dataSplit.main(pathol_train_validate_combo, train_percent, 100 - train_percent, 0)
    

    # =============================================================================
    train_combo    = normal_train_combo    + pathol_train_combo    
    validate_combo = normal_validate_combo + pathol_validate_combo
    test_combo     = normal_test_combo     + pathol_test_combo


    # =============================================================================
    train_package     = loadMelSpectrogram(train_combo,    classes, dsp_package, dataset_path, data, True,  aug_dict)   
    validate_package  = loadMelSpectrogram(validate_combo, classes, dsp_package, dataset_path, data, False, unaug_dict)   
    test_package      = loadMelSpectrogram(test_combo,     classes, dsp_package, dataset_path, data, False, unaug_dict)
    

    # =============================================================================
    train_data,    _,  _, train_label3,    train_dist,    _                       = train_package
    validate_data, _,  _, validate_label3, validate_dist, validate_augment_amount = validate_package
    test_data,     _,  _, test_label3,     test_dist,     test_augment_amount     = test_package
    print(train_dist)
    print(validate_dist)
    print(test_dist)

    
    # =============================================================================
    _, history, encodeLayer_index = autoencoder.main(input_vector_length, train_data, validate_data, arch_bundle, train_bundle_auto)
    best_autoencoder              = load_model(best_model_name)
    best_encoder                  = Model(inputs  = best_autoencoder.inputs, outputs = best_autoencoder.layers[encodeLayer_index].output)
    

    # ===============================================================================
    # save the plot of validation loss
    plt.plot(history.history[monitor])
    plt.savefig(val_loss_plot_name + str(fold_index + 1) + ".png")
    plt.clf()


    # ===============================================================================
    train_data_encoded     = best_encoder.predict(train_data)
    validate_data_encoded  = best_encoder.predict(validate_data)
    test_data_encoded      = best_encoder.predict(test_data)


    # ===============================================================================
    selector               = SelectKBest(chi2, k = 20).fit(train_data_encoded, train_label3)
    train_data_selected    = selector.transform(train_data_encoded)
    validate_data_selected = selector.transform(validate_data_encoded)
    test_data_selected     = selector.transform(test_data_encoded)
    print(train_data_selected.shape, validate_data_selected.shape, test_data_selected.shape)


    # ===============================================================================
    fold_result_package  = mySVM.method1(train_data_selected,    train_label3, 
                                         validate_data_selected, validate_label3, 
                                         test_data_selected,     test_label3,
                                         test_combo,             test_augment_amount)
    
    
    # =============================================================================
    file_acc, file_con_mat, snippet_acc, snippet_con_mat = fold_result_package
    
    total_file_con_mat = total_file_con_mat + file_con_mat
    total_snip_con_mat = total_snip_con_mat + snippet_con_mat
    print(file_acc)
    print(snippet_acc)
    print(file_con_mat)
    print(snippet_con_mat)
    file_results.append(file_acc)
    snippet_results.append(snippet_acc)
    
    
    # =============================================================================
    K.clear_session()
    tf.reset_default_graph()
    os.remove("best_model_this_fold.hdf5")
    # print("Memory Cleared, Saved Model Removed")


    # =============================================================================
    cur_possible_result = ((num_normal - total_file_con_mat[0][1]) / num_normal + (num_pathol - total_file_con_mat[1][0]) / num_pathol) / 2
    print('The best results we can get is:', cur_possible_result)



# =============================================================================
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