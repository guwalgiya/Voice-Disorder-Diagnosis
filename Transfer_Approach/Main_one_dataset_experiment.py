# ===============================================
# Import Packages which needs to tune
import matplotlib
matplotlib.use("Agg")


# ===============================================
# Import Packages and Functions
from   keras                     import backend            as K
from   matplotlib                import pyplot             as plt
from   VGGish_keras              import myVGGish
from   splitData                 import splitData
from   evaluateSVM               import evaluateSVM
from   evaluateCNN               import evaluateCNN
from   keras.models              import Model, load_model
from   getCombination            import getCombination
from   loadVGGishInput           import loadVGGishInput
from   supportVectorMachine      import mySVM
from   sklearn.model_selection   import KFold
import numpy                     as     np
import tensorflow                as     tf
import os
import math
import pickle


# ===============================================
# Stop the terminal printing garbage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ===============================================
# GPU Setting
gpu_taken   = 0.8
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction  = gpu_taken)
sess        = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))


# ===============================================
# Environment
parent_path = "/home/hguan/Voice-Disorder-Diagnosis/Dataset-"
slash       = "/"


# ===============================================
# Dsp Initialization
snippet_hop    = 100
snippet_length = 1000


# ===============================================
# Dsp Initialization
VGGish_load_shape  = (1,  96, 64)
VGGish_train_shape = (96, 64, 1)


# ===============================================
# Dataset Initialization, dataset = Spanish or KayPentax
classes              = ["Normal", "Pathol"]
dataset_name         = "Spanish"
dataset_path         = parent_path     + dataset_name
data_file_name       = "VGGish_Input_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
aug_dict_file_name   = "Dictionary_"   + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"                   
unaug_dict_file_name = "Dictionary_"   + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" 


# ===============================================
# Training / Cross-Validation Initialization
num_folds          = 5   
training_percent   = 90
train_on_augmented = True


# ===============================================
# CNN Training Initialization, metric = "acc" for keras
metric               = "acc"
batch_size           = 64
epoch_limit          = 100000
adam_beta_1          = 0.9
adam_beta_2          = 0.999
learning_rate        = 0.000001
loss_function        = "mean_squared_error"
shuffle_choice       = True
training_verbose     = 1
trainable_list_1     = [True,  True,  True,  True,  True]
trainable_list_2     = [False, False, False, False, False]
CNN_training_package = [learning_rate, epoch_limit, batch_size, metric, shuffle_choice, loss_function, adam_beta_1, adam_beta_2, training_verbose]


# ===============================================
# CNN Callbacks Initialization
callbacks_mode        = "min"
saved_model_name      = "best_model_this_fold.hdf5"
callbacks_monitor     = "val_loss"
callbacks_verbose     = 0
if_only_save_best     = True
callbacks_patience    = 10
val_loss_plot_name    = "Val_Loss_Plot_"
callbacks_min_delta   = 0.0001
CNN_callbacks_package = [saved_model_name, callbacks_mode, callbacks_monitor,  callbacks_patience, callbacks_min_delta, callbacks_verbose, if_only_save_best]
  

# ===============================================
# Result Representation Initialization
file_results_CNN          = []
snippet_results_CNN       = []
total_file_con_mat_CNN    = np.array([[0,0],[0,0]])
total_snippet_con_mat_CNN = np.array([[0,0],[0,0]])


# ===============================================
# Loading Pickle
temp_file_1  = open(dataset_path + slash + data_file_name       + ".pickle", "rb")
temp_file_2  = open(dataset_path + slash + aug_dict_file_name   + ".pickle", "rb")
temp_file_3  = open(dataset_path + slash + unaug_dict_file_name + ".pickle", "rb")


# ===============================================
# Loading data inside Pickles
aug_dict          = pickle.load(temp_file_2)
unaug_dict        = pickle.load(temp_file_3)
VGGish_Input_data = pickle.load(temp_file_1)


# ===============================================
if train_on_augmented:
    train_dict = aug_dict
else:
    train_dict = unaug_dict

# ===============================================
# Load all combos from this dataset, combo = [Name, Class] example: ["WADFJS", "Pathol"]
name_class_combo = np.asarray(getCombination(dataset_path, classes, slash))


# ===============================================
normal_name_class_combo = [x for x in name_class_combo if (x[1] == "Normal")]
pathol_name_class_combo = [x for x in name_class_combo if (x[1] == "Pathol")]


# ===============================================
normal_index_array = np.arange(len(normal_name_class_combo))
pathol_index_array = np.arange(len(normal_name_class_combo), len(name_class_combo))


# ===============================================
kf_spliter = KFold(n_splits = num_folds, shuffle = True)


# ===============================================
normal_split = kf_spliter.split(normal_index_array)
pathol_split = kf_spliter.split(pathol_index_array)


# ===============================================
# Creat N-folds for normal files
normal_split_index = []
for training_validate_index, test_index in normal_split:
    normal_split_index.append([normal_index_array[training_validate_index], normal_index_array[test_index]])


# ===============================================
# Creat N-folds for pathol files
pathol_split_index = [] 
for training_validate_index, test_index in pathol_split:
    pathol_split_index.append([pathol_index_array[training_validate_index], pathol_index_array[test_index]])
    

# ===============================================
# Start to do k-fold Cross Validation
for fold_index in range(num_folds):


    # ===============================================
    print("---> Now Working On Fold ", fold_index + 1,  " ----------------------------")


    # ===============================================
    # For each class, get traininging_validation files and test file
    normal_training_validate_combo = name_class_combo[normal_split_index[fold_index][0]].tolist()
    pathol_training_validate_combo = name_class_combo[pathol_split_index[fold_index][0]].tolist()
    normal_test_combo              = name_class_combo[normal_split_index[fold_index][1]].tolist()
    pathol_test_combo              = name_class_combo[pathol_split_index[fold_index][1]].tolist()


    # ===============================================
    # For each class, split traininging data and validation data
    [normal_training_combo, normal_validate_combo, _] = splitData(normal_training_validate_combo, training_percent, 100 - training_percent, 0)
    [pathol_training_combo, pathol_validate_combo, _] = splitData(pathol_training_validate_combo, training_percent, 100 - training_percent, 0)


    # ===============================================
    # Combine traininging set, validation set, test set
    training_combo = normal_training_combo + pathol_training_combo
    validate_combo = normal_validate_combo + pathol_validate_combo
    test_combo     = normal_test_combo     + pathol_test_combo


    # ===============================================
    # Load all the snippet"s melSpectrograms
    # Training set can use either augmented data or unaugmented data
    # Validation set and test set must use unaugmented data
    training_package = loadVGGishInput(training_combo, classes, VGGish_Input_data, VGGish_load_shape, train_on_augmented, train_dict)  
    validate_package = loadVGGishInput(validate_combo, classes, VGGish_Input_data, VGGish_load_shape, False,              unaug_dict)
    test_package     = loadVGGishInput(test_combo,     classes, VGGish_Input_data, VGGish_load_shape, False,              unaug_dict)
               

    # ===============================================
    # When using this method, we need to use label type 3 which is "Normal" vs "Pathol"
    # Also need type 1 for test data which is [1, 0] vs [0, 1], and type two, which is 0 vs 1
    training_data, training_label_1, training_label_2, training_label_3, training_dist, _                       = training_package
    validate_data, _,                validate_label_2, validate_label_3, validate_dist, validate_augment_amount = validate_package
    test_data,     _,                test_label_2,     test_label_3,     test_dist,     test_augment_amount     = test_package
    

    # ===============================================
    # Rearange tensor's dimension
    training_data = np.moveaxis(training_data, 1, -1)
    validate_data = np.moveaxis(validate_data, 1, -1)
    test_data     = np.moveaxis(test_data,     1, -1)


    # ===============================================
    # Show how many files and snippets in each set
    print(training_dist, training_data.shape)
    print(validate_dist, validate_data.shape)
    print(test_dist,     test_data.shape)
   
    
    # ===============================================
    # Train CNN
    training_history = myVGGish(training_data,           training_label_1, training_label_2, 
                                validate_data,           validate_label_2, 
                                classes,                 "VGGish",
                                VGGish_train_shape,      trainable_list_1, 
                                CNN_training_package,
                                CNN_callbacks_package)
    
    
    # ===============================================
    # Train CNN
    training_history = myVGGish(training_data,           training_label_1, training_label_2, 
                                validate_data,           validate_label_2, 
                                classes,                 "retrained-VGGish",
                                VGGish_train_shape,      trainable_list_2,
                                CNN_training_package,
                                CNN_callbacks_package)

    # ===============================================
    # Load Trained CNN
    fold_best_CNN = load_model(saved_model_name)

    
    # ===============================================
    # save the plot of validation loss
    plt.plot(training_history.history[callbacks_monitor])
    plt.savefig(val_loss_plot_name + str(fold_index + 1) + ".png")
    plt.clf()


    # ===============================================
    # First we release the results directly from the CNN 
    fold_result_package = evaluateCNN(fold_best_CNN, test_combo, test_data, test_label_3, test_augment_amount, classes)


    # ===============================================
    # Unpack the evaluation result
    fold_file_acc, fold_file_con_mat, fold_snippet_acc, fold_snippet_con_mat = fold_result_package

    
    # ===============================================
    # Print the result for this fold 
    print("Now we have results directly from the CNN:")
    print("The file macro accuracy for this fold is:    ", fold_file_acc)
    print("The snippet macro accuracy for this fold is: ", fold_snippet_acc)
    print("File confusion matrix for this fold is: ")
    print(fold_file_con_mat)
    print("Snippet confusion matrix for this fold is:")
    print(fold_snippet_con_mat)


    # ===============================================
    # Update overall results
    file_results_CNN.append(fold_file_acc)
    snippet_results_CNN.append(fold_snippet_acc)


    # ===============================================
    # Update overall confusion matrix
    total_file_con_mat_CNN    = total_file_con_mat_CNN    + fold_file_con_mat
    total_snippet_con_mat_CNN = total_snippet_con_mat_CNN + fold_snippet_con_mat


# ===============================================
# Show Final Results after cross-validation
# Classification Accuracy for each fold (file level)
print("Now showing the result if we use CNN directly: ")
print("--------------------------------")
print("file results")
print(file_results_CNN)
print(sum(file_results_CNN) / len(file_results_CNN))


# ===============================================
# Show Final Results after cross-validation
# Classification Accuracy for each fold (snippet level)
print("--------------------------------")
print("snippet results")
print(snippet_results_CNN)
print(sum(snippet_results_CNN) / len(snippet_results_CNN))


# ===============================================
# Macro Accuracy for the whole experiment (file level)
print("--------------------------------")
print("final file results")
print(total_file_con_mat_CNN)


# ===============================================
file_overall_acc = 0;
for i in range(len(total_file_con_mat_CNN[0])):
    file_overall_acc = file_overall_acc + total_file_con_mat_CNN[i][i] / sum(total_file_con_mat_CNN[i])
print(file_overall_acc / len(classes))


# ===============================================
# Macro Accuracy for the whole experiment (snippet level)
print("--------------------------------")
print("final snippet results")
print(total_snippet_con_mat_CNN)


# ===============================================
snippet_overall_acc = 0;
for i in range(len(total_snippet_con_mat_CNN[0])):
    snippet_overall_acc = snippet_overall_acc + total_snippet_con_mat_CNN[i][i] / sum(total_snippet_con_mat_CNN[i])
print(snippet_overall_acc / len(classes)) 