# ===============================================
# Import Packages and Functions
from   sklearn.model_selection import KFold
from   supportVectorMachine    import mySVM
from   getCombination          import getCombination
from   evaluateSVM             import evaluateSVM
from   splitData               import splitData
from   loadMFCCs               import loadMFCCs
import numpy                   as     np
import pickle


# ===============================================
# Environment
parent_path = "/home/hguan/7100-Master-Project/Dataset-"
slash       = "/"


# ===============================================
# Dsp Initialization, num_features = num_of_aggregated_MFCCs
fs             = 16000
fft_hop        = 128
fft_length     = 512
snippet_hop    = 100
num_features   = 40
snippet_length = 500


# ===============================================
# Dataset Initialization, dataset = Spanish or KayPentax
classes              = ["Normal", "Pathol"]
dataset_name         = "KayPentax"
dataset_path         = parent_path   + dataset_name
data_file_name       = "MFCCs_"      + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_block" + str(fft_length) + "_hop" + str(fft_hop)
aug_dict_file_name   = "Dictionary_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" 
unaug_dict_file_name = "Dictionary_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" 


# ===============================================
# Cross-Validation Initialization
num_folds        = 5
training_percent = 90


# ===============================================
# SVM Initialization
c_values             = [0.1, 1, 10, 100, 1000]
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
temp_file_1  = open(dataset_path + slash + data_file_name       + '.pickle', 'rb')  
temp_file_2  = open(dataset_path + slash + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3  = open(dataset_path + slash + unaug_dict_file_name + '.pickle', 'rb')


# ===============================================
# Loading data inside Pickles
aug_dict   = pickle.load(temp_file_2)
unaug_dict = pickle.load(temp_file_3)
MFCCs_data = pickle.load(temp_file_1)


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
    # Load all the snippet's aggregated MFCCs
    # Training set can use either augmented data or unaugmented data
    # Validation set and test set must use unaugmented data
    training_package  = loadMFCCs(training_combo, classes, num_features, MFCCs_data, True,  aug_dict)   
    validate_package  = loadMFCCs(validate_combo, classes, num_features, MFCCs_data, False, unaug_dict)  
    test_package      = loadMFCCs(test_combo,     classes, num_features, MFCCs_data, False, unaug_dict)
    

    # ===============================================
    # When using this method, we need to use label type 3 which is "Normal" vs "Pathol"
    training_data, _,    _, training_label_3,  training_dist, _                   = training_package
    validate_data, _,    _, validate_label_3,  validate_dist, _                   = validate_package
    test_data,     _,    _, test_label_3,      test_dist,     test_augment_amount = test_package
    

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
    for i in range(num_features):  


        # ===============================================
        # Mean value and standard deviation is computed from the training and validation sets
        cur_max = np.max(training_data[:, i].flatten().tolist() + validate_data[:, i].flatten().tolist())
        cur_min = np.min(training_data[:, i].flatten().tolist() + validate_data[:, i].flatten().tolist())


        # ===============================================
        # Normalization
        training_data_normalized[:, i] = (training_data[:, i] - cur_min)  / (cur_max - cur_min)
        validate_data_normalized[:, i] = (validate_data[:, i] - cur_min)  / (cur_max - cur_min)
        test_data_normalized[:, i]     = (test_data[:, i]     - cur_min)  / (cur_max - cur_min)
        

        # ===============================================
        # For test set, we need to handle values not in the range of [0,1]
        np.clip(test_data_normalized[:, i], 0, 1)


    # ===============================================
    # Train SVMs, search the best parameters
    fold_best_SVM = mySVM(training_data_normalized, training_label_3,
    	                  validate_data_normalized, validate_label_3,
    	                  svm_training_package,     classes)

      
    # ===============================================
    # Evaluate the best svm for this fold, using test data
    fold_result_package = evaluateSVM(fold_best_SVM,        test_combo, 
    	                              test_data_normalized, test_label_3, 
    	                              test_augment_amount,  classes)


    # ===============================================
    # Break the result package
    cur_file_acc, cur_file_con_mat, cur_snippet_acc, cur_snippet_con_mat = fold_result_package
    

    # ===============================================
    # Print the result for this fold
    print("The file macro accuracy for this fold is:    ", cur_file_acc)
    print("The snippet macro accuracy for this fold is: ", cur_snippet_acc)
    print("File confusion matrix for this fold is:")
    print(cur_file_con_mat)
    print("Snippet confusion matrix for this fold is:")
    print(cur_snippet_con_mat)


    # ===============================================
    # Update overall results
    file_results.append(cur_file_acc)
    snippet_results.append(cur_snippet_acc)


    # ===============================================
    # Update overall confusion matrix
    total_file_con_mat    = total_file_con_mat    + cur_file_con_mat
    total_snippet_con_mat = total_snippet_con_mat + cur_snippet_con_mat


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
print('snippet results')
print(snippet_results)
print(sum(snippet_results) / len(snippet_results))


# ===============================================
# Macro Accuracy for the whole experiment (file level)
print("--------------------------------")
print('final file results')
print(total_file_con_mat)


# ===============================================
file_overall_acc = 0;
for i in range(len(total_file_con_mat[0])):
    file_overall_acc = file_overall_acc + total_file_con_mat[i][i] / sum(total_file_con_mat[i])
print(file_overall_acc / len(classes))


# ===============================================
# Macro Accuracy for the whole experiment (snippet level)
print('--------------------------------')
print('final snippet results')
print(total_snippet_con_mat)


# ===============================================
snippet_overall_acc = 0;
for i in range(len(total_snippet_con_mat[0])):
    snippet_overall_acc = snippet_overall_acc + total_snippet_con_mat[i][i] / sum(total_snippet_con_mat[i])
print(snippet_overall_acc / len(classes))    