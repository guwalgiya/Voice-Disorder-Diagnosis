# ===============================================
# Import Packages which needs to tune
import matplotlib
matplotlib.use("Agg")


# ===============================================
# Import Packages and Functions
from   keras                     import backend            as K
from   matplotlib                import pyplot             as plt
from   CNN                       import myCNN
from   splitData                 import splitData
from   evaluateSVM               import evaluateSVM
from   evaluateCNN               import evaluateCNN
from   keras.models              import Model, load_model
from   getCombination            import getCombination
from   loadMelSpectrogram        import loadMelSpectrogram
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
gpu_taken   = 0.4
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction  = gpu_taken)
sess        = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))


# ===============================================
# Environment
parent_path = "/home/hguan/7100-Master-Project/Dataset-"
slash       = "/"


# ===============================================
# Dsp Initialization, num_rows = num_MFCCs (not aggregated)
fs              = 16000
fft_hop         = 128
fft_length      = 512
mel_length      = 128
snippet_hop     = 100
snippet_length  = 500
num_time_frames = math.ceil(snippet_length / 1000 * fs / fft_hop)


# ===============================================
# Dataset Initialization, dataset = Spanish or KayPentax
classes              = ["Normal", "Pathol"]
input_type           = "MelSpectrogram"
dataset_name         = "KayPentax"
dataset_path         = parent_path       + dataset_name
data_file_name       = "MelSpectrogram_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop) + "_mel" + str(mel_length)
aug_dict_file_name   = "Dictionary_"     + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"                   
unaug_dict_file_name = "Dictionary_"     + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" 


# ===============================================
# Cross-Validation Initialization
num_folds        = 5   
training_percent = 90


# ===============================================
# CNN Architecture Initialization
num_channel              = 4
input_shape              = (int(mel_length / num_channel), math.ceil(snippet_length / 1000 * fs / fft_hop), num_channel)
FC_num_neuron_list       = [1024, 512, 256, 128, 64, 32, 16, 8, 4]
CNN_architecture_package = [input_shape, FC_num_neuron_list]


# ===============================================
# CNN Training Initialization, metric = "acc" for keras
metric               = "acc"
batch_size           = 4096
epoch_limit          = 10000000
adam_beta_1          = 0.9
adam_beta_2          = 0.999
learning_rate        = 0.0001
loss_function        = "mean_squared_error"
shuffle_choice       = True
training_verbose     = 0
CNN_training_package = [learning_rate, epoch_limit, batch_size, metric, shuffle_choice, loss_function, adam_beta_1, adam_beta_2, training_verbose]


# ===============================================
# CNN Callbacks Initialization
callbacks_mode        = "min"
saved_model_name      = "best_model_this_fold.hdf5"
callbacks_monitor     = "val_loss"
callbacks_verbose     = 0
if_only_save_best     = True
callbacks_patience    = 30
val_loss_plot_name    = "Val_Loss_Plot_"
callbacks_min_delta   = 0.0001
CNN_callbacks_package = [saved_model_name, callbacks_mode, callbacks_monitor,  callbacks_patience, callbacks_min_delta, callbacks_verbose, if_only_save_best]
  

# ===============================================
# SVM Initialization
c_values             = [0.1, 1, 10, 100]
svm_verbose          = 0
svm_tolerance        = 0.001
svm_max_iteration    = 1000
svm_training_package = [c_values, svm_verbose, svm_tolerance, svm_max_iteration]



# ===============================================
# Result Representation Initialization
file_results_CNN          = []
snippet_results_CNN       = []
total_file_con_mat_CNN    = np.array([[0,0],[0,0]])
total_snippet_con_mat_CNN = np.array([[0,0],[0,0]])


# ===============================================
# Result Representation Initialization
file_results_SVM          = []
snippet_results_SVM       = []
total_file_con_mat_SVM    = np.array([[0,0],[0,0]])
total_snippet_con_mat_SVM = np.array([[0,0],[0,0]])


# ===============================================
# Loading Pickle
temp_file_1  = open(dataset_path + slash + data_file_name       + ".pickle", "rb")  
temp_file_2  = open(dataset_path + slash + aug_dict_file_name   + ".pickle", "rb")
temp_file_3  = open(dataset_path + slash + unaug_dict_file_name + ".pickle", "rb")


# ===============================================
# Loading data inside Pickles
aug_dict            = pickle.load(temp_file_2)
unaug_dict          = pickle.load(temp_file_3)
melSpectrogram_data = pickle.load(temp_file_1)


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
    training_package = loadMelSpectrogram(training_combo, classes, mel_length, num_time_frames, input_type, melSpectrogram_data, False, unaug_dict)   
    validate_package = loadMelSpectrogram(validate_combo, classes, mel_length, num_time_frames, input_type, melSpectrogram_data, False, unaug_dict)   
    test_package     = loadMelSpectrogram(test_combo,     classes, mel_length, num_time_frames, input_type, melSpectrogram_data, False, unaug_dict)
    

    # ===============================================
    # When using this method, we need to use label type 3 which is "Normal" vs "Pathol"
    # Also need type 1 for test data which is [1, 0] vs [0, 1], and type two, which is 0 vs 1
    training_data, training_label_1, training_label_2, training_label_3, training_dist, _                       = training_package
    validate_data, _,                validate_label_2, validate_label_3, validate_dist, validate_augment_amount = validate_package
    test_data,     _,                test_label_2,     test_label_3,     test_dist,     test_augment_amount     = test_package
    

    # ===============================================
    # Show how many files and snippets in each set
    print(training_dist)
    print(validate_dist)
    print(test_dist)


    # ===============================================
    # Change the one-channel melspectrogram to multiple-channels
    training_data = training_data.reshape(training_data.shape[0], num_channel, int(training_data.shape[1] / num_channel), training_data.shape[2])   
    validate_data = validate_data.reshape(validate_data.shape[0], num_channel, int(validate_data.shape[1] / num_channel), validate_data.shape[2]) 
    test_data     = test_data.reshape(test_data.shape[0],         num_channel, int(test_data.shape[1]     / num_channel), test_data.shape[2]) 
    
    
    # ===============================================
    # Rearange tensor"s dimension
    training_data = np.moveaxis(training_data, 1, -1)
    validate_data = np.moveaxis(validate_data, 1, -1)
    test_data     = np.moveaxis(test_data,     1, -1)


    # ===============================================
    # Train CNN
    training_history = myCNN(training_data,           training_label_1, training_label_2, 
                             validate_data,           validate_label_2, 
                             classes,
                             CNN_architecture_package,
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
    # Start to select the best dimension for this fold
    fold_best_file_acc = 0
    for dimension in FC_num_neuron_list:


        # ===============================================
        # Form a feature extractor from the trained CNN      
        index         = FC_num_neuron_list.index(dimension)
        cur_extractor = Model(inputs  = fold_best_CNN.inputs, 
                              outputs = fold_best_CNN.layers[-1 - index].output)

        
        # ===============================================
        # Get encoded training data and validate data  
        training_data_CNNed = cur_extractor.predict(training_data)
        validate_data_CNNed = cur_extractor.predict(validate_data)
        
        
        # ===============================================
        # Train SVMs, search the best parameters
        cur_SVM = mySVM(training_data_CNNed,  training_label_3,
                        validate_data_CNNed,  validate_label_3,
                        svm_training_package, classes)
        
        
        # ===============================================
        # Check how good is this encoding dimension perform on the validation set
        cur_result_package = evaluateSVM(cur_SVM,                 validate_combo, 
                                         validate_data_CNNed,     validate_label_3, 
                                         validate_augment_amount, classes)
        

        # ===============================================
        cur_file_acc, _, _, _, = cur_result_package
        

        # ===============================================
        # if current result is good enough on the validation set
        # Then we keep current encoder + SVM as our best model combination
        if cur_file_acc         > fold_best_file_acc:
            fold_best_SVM       = cur_SVM
            fold_best_file_acc  = cur_file_acc
            fold_best_dimension = dimension
            fold_best_extractor = cur_extractor

   
    # ===============================================
    # Print dimension searching result
    print("For this fold, the best encoder's dimension is: ", fold_best_dimension)

     
    # ===============================================
    # Prepare "CNNed" test set
    test_data_CNNed = fold_best_extractor.predict(test_data)


    # ===============================================
    # Test our best model combination
    fold_result_package = evaluateSVM(fold_best_SVM,       test_combo, 
                                      test_data_CNNed,     test_label_3, 
                                      test_augment_amount, classes)


    # ===============================================
    # Unpack the evaluation result
    fold_file_acc, fold_file_con_mat, fold_snippet_acc, fold_snippet_con_mat = fold_result_package
    
    
    # ===============================================
    # Print the result for this fold
    print("After searching the best extractor, we have the following result: ")
    print("The file macro accuracy for this fold is:    ", fold_file_acc)
    print("The snippet macro accuracy for this fold is: ", fold_snippet_acc)
    print("File confusion matrix for this fold is:")
    print(fold_file_con_mat)
    print("Snippet confusion matrix for this fold is:")
    print(fold_snippet_con_mat)


    # ===============================================
    # Update overall results
    file_results_SVM.append(fold_file_acc)
    snippet_results_SVM.append(fold_snippet_acc)


    # ===============================================
    # Update overall confusion matrix
    total_file_con_mat_SVM    = total_file_con_mat_SVM    + fold_file_con_mat
    total_snippet_con_mat_SVM = total_snippet_con_mat_SVM + fold_snippet_con_mat



    # ===============================================
    # Clean almost everything for last fold, otherwise computer might crash
    K.clear_session()
    tf.reset_default_graph()
    os.remove(saved_model_name)
    print("Memory Cleared, Saved Model Removed")


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


# ===============================================
# Show Final Results after cross-validation
# Classification Accuracy for each fold (file level)
print("Now showing the result if we search CNN"s layers: ")
print("--------------------------------")
print("file results")
print(file_results_SVM)
print(sum(file_results_SVM) / len(file_results_SVM))


# ===============================================
# Show Final Results after cross-validation
# Classification Accuracy for each fold (snippet level)
print("--------------------------------")
print("snippet results")
print(snippet_results_SVM)
print(sum(snippet_results_SVM) / len(snippet_results_SVM))


# ===============================================
# Macro Accuracy for the whole experiment (file level)
print("--------------------------------")
print("final file results")
print(total_file_con_mat_SVM)


# ===============================================
file_overall_acc = 0;
for i in range(len(total_file_con_mat_SVM[0])):
    file_overall_acc = file_overall_acc + total_file_con_mat_SVM[i][i] / sum(total_file_con_mat_SVM[i])
print(file_overall_acc / len(classes))


# ===============================================
# Macro Accuracy for the whole experiment (snippet level)
print("--------------------------------")
print("final snippet results")
print(total_snippet_con_mat_SVM)


# ===============================================
snippet_overall_acc = 0;
for i in range(len(total_snippet_con_mat_SVM[0])):
    snippet_overall_acc = snippet_overall_acc + total_snippet_con_mat_SVM[i][i] / sum(total_snippet_con_mat_SVM[i])
print(snippet_overall_acc / len(classes)) 