# ===============================================
# Import Packages which needs to tune
import matplotlib
matplotlib.use("Agg")


# ===============================================
# Import Packages and Functions
from   keras                     import backend            as K
from   matplotlib                import pyplot             as plt
from   splitData                 import splitData
from   autoencoder               import myAutoencoder
from   evaluateSVM               import evaluateSVM
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
num_rows        = 20
fft_length      = 512
mel_length      = 128
snippet_hop     = 100
snippet_length  = 500
num_time_frames = math.ceil(snippet_length / 1000 * fs / fft_hop)


# ===============================================
# Dataset Initialization, dataset = Spanish or KayPentax
classes              = ["Normal", "Pathol"]
input_type           = "MFCCs"
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
# Autoencoder Architecture Initialization
encoding_dimension      = 32
num_encoding_layer      = 5
num_decoding_layer      = 5
input_vector_length     = num_rows * math.ceil(snippet_length / 1000 * fs / fft_hop)
AE_architecture_package = [input_vector_length, encoding_dimension, num_encoding_layer, num_decoding_layer]


# ===============================================
# Autoencoder Training Initialization
batch_size          = 4096
epoch_limit         = 1000000
adam_beta_1         = 0.9
adam_beta_2         = 0.999
learning_rate       = 0.0001
loss_function       = "mean_squared_error"
shuffle_choice      = True
training_verbose    = 0
AE_training_package = [learning_rate, epoch_limit, batch_size, shuffle_choice, loss_function, adam_beta_1, adam_beta_2, training_verbose]


# ===============================================
# Autoencoder Callbacks Initialization
callbacks_mode       = "min"
saved_model_name     = "best_model_this_fold.hdf5"
callbacks_monitor    = "val_loss"
callbacks_verbose    = 0
if_only_save_best    = True
callbacks_patience   = 30
val_loss_plot_name   = "Val_Loss_Plot_"
callbacks_min_delta  = 0.0001
AE_callbacks_package = [saved_model_name, callbacks_mode, callbacks_monitor,  callbacks_patience, callbacks_min_delta, callbacks_verbose, if_only_save_best]
   

# ===============================================
# SVM Initialization
c_values             = [0.1, 1, 10, 100]
svm_verbose          = 0
svm_tolerance        = 0.001
svm_max_iteration    = 1000
svm_training_package = [c_values, svm_verbose, svm_tolerance, svm_max_iteration]


# ===============================================
# Result Representation Initialization
file_results          = []
snippet_results       = []
total_file_con_mat    = np.array([[0,0],[0,0]])
total_snippet_con_mat = np.array([[0,0],[0,0]])


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
    training_package = loadMelSpectrogram(training_combo, classes, num_rows, num_time_frames, input_type, melSpectrogram_data, False,  unaug_dict)   
    validate_package = loadMelSpectrogram(validate_combo, classes, num_rows, num_time_frames, input_type, melSpectrogram_data, False, unaug_dict)   
    test_package     = loadMelSpectrogram(test_combo,     classes, num_rows, num_time_frames, input_type, melSpectrogram_data, False, unaug_dict)
    

    # ===============================================
    # When using this method, we need to use label type 3 which is "Normal" vs "Pathol"
    training_data, _, _, training_label_3, training_dist, _                       = training_package
    validate_data, _, _, validate_label_3, validate_dist, validate_augment_amount = validate_package
    test_data,     _, _, test_label_3,     test_dist,     test_augment_amount     = test_package
    

    # ===============================================
    # Show how many files and snippets in each set
    print(training_dist)
    print(validate_dist)
    print(test_dist)
    

    # ===============================================
    # Normalization - allocating memory
    training_data_normalized = np.zeros((training_data.shape))
    validate_data_normalized = np.zeros((validate_data.shape))
    test_data_normalized     = np.zeros((test_data.shape))


    # ===============================================
    # For each feature, we normalize the whole dataset 
    for i in range(num_rows):


        # ===============================================
        # Sample Mean value and standard deviation is computed from the training and validation sets
        mean = np.mean(training_data[:, i, :].flatten().tolist() + validate_data[:, i, :].flatten().tolist())
        std  = np.std(training_data[:, i, :].flatten().tolist()  + validate_data[:, i, :].flatten().tolist())
        

        # ===============================================
        # Normalization
        training_data_normalized[:, i, :]  = (training_data[:, i, :] - mean) / std
        validate_data_normalized[:, i, :]  = (validate_data[:, i, :] - mean) / std
        test_data_normalized[:, i, :]      = (test_data[:, i, :]     - mean) / std
        

        # ===============================================
        # For test set, we need to handle values not in the range of [0,1]
        np.clip(test_data_normalized[:, i, :], 0, 1)


    # ===============================================
    # Reshape
    training_data_normalized = training_data_normalized.reshape((len(training_data_normalized), np.prod(training_data_normalized.shape[1:])), order = "F") 
    validate_data_normalized = validate_data_normalized.reshape((len(validate_data_normalized), np.prod(validate_data_normalized.shape[1:])), order = "F")
    test_data_normalized     = test_data_normalized.reshape((len(test_data_normalized),         np.prod(test_data_normalized.shape[1:])),     order = "F")
 

    # ===============================================
    # Train Autoencoder
    training_history, encoding_layer_index = myAutoencoder(training_data_normalized, validate_data_normalized, AE_architecture_package, AE_training_package, AE_callbacks_package)
    

    # ===============================================
    # Load Trained Autoencoder   
    fold_best_AE = load_model(saved_model_name)


    # ===============================================
    # Plot Validation loss
    plt.plot(training_history.history[callbacks_monitor])
    plt.savefig(val_loss_plot_name + str(fold_index + 1) + ".png")
    plt.clf()


    # ===============================================
    # Now we want to know which hidden layer is the best
    # We creat a list of possible dimensions
    dimension_choices = [encoding_dimension * 2 ** x for x in range(num_encoding_layer + 1)]
    
    
    # ===============================================
    # Start to select the best dimension for this fold
    fold_best_file_acc = 0
    for dimension in dimension_choices:
        
        
        # ===============================================
        # Form a encoder from the trained autoencoder
        index       = dimension_choices.index(dimension)
        cur_encoder = Model(inputs  = fold_best_AE.input, 
                            outputs = fold_best_AE.layers[encoding_layer_index - index].output)
        
        
        # ===============================================
        # Get encoded training data and validate data      
        training_data_encoded = cur_encoder.predict(training_data_normalized)
        validate_data_encoded = cur_encoder.predict(validate_data_normalized)
        

        # ===============================================
        # Train SVMs, search the best parameters
        cur_SVM = mySVM(training_data_encoded, training_label_3,
                        validate_data_encoded, validate_label_3,
                        svm_training_package,  classes)


        # ===============================================
        # Check how good is this encoding dimension perform on the validation set
        cur_result_package = evaluateSVM(cur_SVM,                 validate_combo, 
    	                                 validate_data_encoded,   validate_label_3, 
    	                                 validate_augment_amount, classes)
 
        
        # ===============================================
        cur_file_acc, _, _, _, = cur_result_package
        
        
        # ===============================================
        # if current result is good enough on the validation set
        # Then we keep current encoder + SVM as our best model combination
        if cur_file_acc         > fold_best_file_acc:
            fold_best_SVM       = cur_SVM
            fold_best_encoder   = cur_encoder
            fold_best_file_acc  = cur_file_acc
            fold_best_dimension = dimension


    # ===============================================
    # Print dimension searching result
    print("For this fold, the best encoder's dimension is: ", fold_best_dimension)
    print(fold_best_encoder.summary())
    
    
    # ===============================================
    # Prepare test set
    test_data_encoded = fold_best_encoder.predict(test_data_normalized) 
    
    
    # ===============================================
    # Test our best model combination
    fold_result_package = evaluateSVM(fold_best_SVM,       test_combo, 
	                                  test_data_encoded,   test_label_3, 
	                                  test_augment_amount, classes)
    
    
    # ===============================================
    # Unpack the evaluation result
    fold_file_acc, fold_file_con_mat, fold_snippet_acc, fold_snippet_con_mat = fold_result_package
    
    
    # ===============================================
    # Print the result for this fold
    print("The file macro accuracy for this fold is:    ", fold_file_acc)
    print("The snippet macro accuracy for this fold is: ", fold_snippet_acc)
    print("File confusion matrix for this fold is:")
    print(fold_file_con_mat)
    print("Snippet confusion matrix for this fold is:")
    print(fold_snippet_con_mat)


    # ===============================================
    # Update overall results
    file_results.append(fold_file_acc)
    snippet_results.append(fold_snippet_acc)


    # ===============================================
    # Update overall confusion matrix
    total_file_con_mat    = total_file_con_mat    + fold_file_con_mat
    total_snippet_con_mat = total_snippet_con_mat + fold_snippet_con_mat
    
    
    # ===============================================
    # Clean almost everything for last fold, otherwise computer might crash
    K.clear_session()
    tf.reset_default_graph()
    os.remove(saved_model_name)
    print("Memory Cleared, Saved Model Removed")


# ===============================================
# Show Final Results after cross-validation
# Classification Accuracy for each fold (file level)
print("--------------------------------")
print("file results")
print(file_results)
print(sum(file_results) / len(file_results))


# ===============================================
# Show Final Results after cross-validation
# Classification Accuracy for each fold (snippet level)
print("--------------------------------")
print("snippet results")
print(snippet_results)
print(sum(snippet_results) / len(snippet_results))


# ===============================================
# Macro Accuracy for the whole experiment (file level)
print("--------------------------------")
print("final file results")
print(total_file_con_mat)


# ===============================================
file_overall_acc = 0;
for i in range(len(total_file_con_mat[0])):
    file_overall_acc = file_overall_acc + total_file_con_mat[i][i] / sum(total_file_con_mat[i])
print(file_overall_acc / len(classes))


# ===============================================
# Macro Accuracy for the whole experiment (snippet level)
print("--------------------------------")
print("final snippet results")
print(total_snippet_con_mat)


# ===============================================
snippet_overall_acc = 0;
for i in range(len(total_snippet_con_mat[0])):
    snippet_overall_acc = snippet_overall_acc + total_snippet_con_mat[i][i] / sum(total_snippet_con_mat[i])
print(snippet_overall_acc / len(classes)) 