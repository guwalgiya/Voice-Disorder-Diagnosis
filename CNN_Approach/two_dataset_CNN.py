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
from   math                    import ceil

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction  = 0.4)
sess        = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =============================================================================
# Dataset Initialization
classes = ["Normal", "Pathol"]
dataset_path_train  = "/home/hguan/7100-Master-Project/Dataset-Spanish"
dataset_path_test   = "/home/hguan/7100-Master-Project/Dataset-KayPentax"
train_percent            = 90
validate_percent         = 10
test_percent             = 0
best_model_name    = "best_model_this_fold.hdf5"
val_loss_plot_name                 = "Val_Loss_Plot_"

# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500   #in milliseconds
snippet_hop         = 100   #in ms
fft_length          = 512
fft_hop             = 128
mel_length          = 128
num_MFCCs           = 20
num_rows            = mel_length
dsp_package         = [fs, snippet_length, snippet_hop, fft_length, fft_hop]
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
data_file_name       = input_name + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
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
validate_combo = normal_validate_combo + pathol_validate_combo

# =============================================================================
test_combo          = getCombination.main(dataset_path_test,  classes)
[_, _, test_combo]  = dataSplit.main(test_combo, 0, 0, 100)


# =============================================================================
train_package     = loadMelSpectrogram(train_combo,    classes, dsp_package, num_rows, "MelSpec", data_1, True,  aug_dict_1)   
validate_package  = loadMelSpectrogram(validate_combo, classes, dsp_package, num_rows, "MelSpec", data_1, False, unaug_dict_1)   
test_package      = loadMelSpectrogram(test_combo,     classes, dsp_package, num_rows, "MelSpec", data_2, False, unaug_dict_2)


# =============================================================================
train_data,    _,            train_label_2,    train_label_3,    train_dist,    _                   = train_package
validate_data, _,            validate_label_2, validate_label_3, validate_dist, _                   = validate_package
test_data,     test_label_1, test_label_2,     test_label_3,     test_dist,     test_augment_amount = test_package
print(train_dist)
print(validate_dist)
print(test_dist)


# ==============================================================================
train_data    = train_data.reshape(train_data.shape[0],       num_channel, int(train_data.shape[1]    / num_channel), train_data.shape[2])   
validate_data = validate_data.reshape(validate_data.shape[0], num_channel, int(validate_data.shape[1] / num_channel), validate_data.shape[2]) 
test_data     = test_data.reshape(test_data.shape[0],         num_channel, int(test_data.shape[1]     / num_channel), test_data.shape[2]) 


# ==============================================================================
train_data    = np.moveaxis(train_data,    1, -1)
validate_data = np.moveaxis(validate_data, 1, -1)
test_data     = np.moveaxis(test_data,     1, -1)


# ==============================================================================
_, history = CNN.main(train_data, train_label_2, train_label_3, test_data, test_label_2, test_label_3, epoch_limit, batch_size, input_shape, monitor)
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
                                         test_data_CNNed,     test_label_3, 
                                         test_data_CNNed,     test_label_3,
                                         test_combo,          test_augment_amount)

    cur_file_acc, cur_file_con_mat, cur_snippet_acc, cur_snippet_con_mat = fold_result_package
    print(dim, cur_file_acc)
    if cur_file_acc > best_file_acc:
        best_file_acc    = cur_file_acc
        best_result_pack = fold_result_package
        best_dim         = dim


# =============================================================================
file_acc, file_con_mat, snippet_acc, snippet_con_mat = best_result_pack
print('----------------------------')
print(file_acc)
print(snippet_acc)
print(file_con_mat)
print(snippet_con_mat)


# ==============================================================================
K.clear_session()
tf.reset_default_graph()
os.remove("best_model_this_fold.hdf5")
print("Memory Cleared, Saved Model Removed")


# =============================================================================
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