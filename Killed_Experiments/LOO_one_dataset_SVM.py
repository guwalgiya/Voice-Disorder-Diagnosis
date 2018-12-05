# =============================================================================
# Import Packages
from   sklearn.model_selection import KFold
from   sklearn.utils           import class_weight
from   loadMFCCs               import loadMFCCs
from   sklearn                 import svm
import numpy                   as     np
import math
import pickle
import dataSplit
import getCombination
import resultsAnalysisSVM
import resultsAnalysisSVMOneFile


# =============================================================================
# Machine Learning Initialization
dataset_path  = "/home/hguan/7100-Master-Project/Dataset-KayPentax"
classes       = ['Normal','Pathol']
num_folds     = 655 + 53
c_values      = [0.1, 1, 10, 100]
svm_verbose   = 0     
train_percent = 90
input_name    = "MFCCs"


# =============================================================================
# Dsp Initialization
fs                  = 16000
snippet_length      = 500   # in milliseconds
snippet_hop         = 100   # in ms
fft_length          = 512
fft_hop             = 128
num_features        = 40    # 40 features in this approach
dsp_package         = [fs, snippet_length, snippet_hop, fft_length, fft_hop, num_features]


# =============================================================================
# Leave One Out
name_class_combo = getCombination.main(dataset_path, classes)
name_class_combo = np.asarray(name_class_combo)
kf_spliter       = KFold(n_splits = num_folds, shuffle = True)
index_array      = np.arange(len(name_class_combo))
file_num         = 1


# ==============================================================================
file_results     = []
snippet_results  = []
file_con_mat     = np.array([[0,0],[0,0]])
snippet_con_mat  = np.array([[0,0],[0,0]])


# ==============================================================================
try:
    temp_file_0         = open(dataset_path + "/" + "one_dataset_baseline_results.pickle", "rb" )
    results             = pickle.load(temp_file_0)
    file_results, snippet_results, file_con_mat, snippet_con_mat = results["Results"]
except:
    results             = {}
    results["Results"]  = [file_results, snippet_results, file_con_mat, snippet_con_mat]


# ==============================================================================
data_file_name       = input_name + "_" + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" +                  "_block" + str(fft_length) + "_hop" + str(fft_hop) 
aug_dict_file_name   = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms"
unaug_dict_file_name = "Dictionary_"    + str(snippet_length) + "ms_" + str(snippet_hop) + "ms" + "_unaugmented" 

temp_file_1          = open(dataset_path + '/' + data_file_name       + '.pickle', 'rb')  
temp_file_2          = open(dataset_path + '/' + aug_dict_file_name   + '.pickle', 'rb')
temp_file_3          = open(dataset_path + '/' + unaug_dict_file_name + '.pickle', 'rb')
data                 = pickle.load(temp_file_1)
aug_dict             = pickle.load(temp_file_2)
unaug_dict           = pickle.load(temp_file_3)


# ==============================================================================
# start to do outer k-folds cross validation
for train_validate_index, test_index in kf_spliter.split(index_array):
    print('---> File ', file_num, ' ---------------------------------')
    
    # ==============================================================================
    test_combo   = name_class_combo[test_index]    
    test_combo   = test_combo.tolist()
    test_name    = test_combo[0][0]
    test_class   = test_combo[0][1]


    # ==============================================================================
    if test_name in results.keys():
        print("We already did ", test_name)
    else:


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
        train_package    = loadMFCCs(train_combo,    classes, dsp_package, dataset_path, data, True,  aug_dict)
        validate_package = loadMFCCs(validate_combo, classes, dsp_package, dataset_path, data, False, unaug_dict)

        train_data,    _,  _, train_label_3,    train_dist,    _                       = train_package
        validate_data, _,  _, validate_label_3, validate_dist, validate_augment_amount = validate_package
        

        # ==============================================================================
        train_data_normalized    = np.zeros((train_data.shape))
        validate_data_normalized = np.zeros((validate_data.shape))
        
        standard_max_list  = [0] * num_features
        standard_min_list  = [0] * num_features


        for i in range(num_features):

            # ==============================================================================
            standard_max = max(np.amax(train_data[:, i]), np.amax(validate_data[:, i]))
            standard_min = min(np.amin(train_data[:, i]), np.amin(validate_data[:, i]))

            standard_max_list[i] = round(standard_max, 3)
            standard_min_list[i] = round(standard_min, 3)
               
            train_data_normalized[:, i]    = (train_data[:, i]    - standard_min_list[i]) / (standard_max_list[i] - standard_min_list[i])
            validate_data_normalized[:, i] = (validate_data[:, i] - standard_min_list[i]) / (standard_max_list[i] - standard_min_list[i])
        

        # ============================================================================== 
        best_c          = 0    
        best_search_acc = 0
        for cur_c in c_values:
            
            
            # ==============================================================================
            train_class_weight_raw = class_weight.compute_class_weight('balanced', np.unique(train_label_3), train_label_3)
            train_class_weight     = {}
            for a_class in np.unique(train_label_3):
                train_class_weight[a_class] = train_class_weight_raw[np.unique(train_label_3).tolist().index(a_class)]


            # ==============================================================================
            cur_svm = svm.LinearSVC(C = cur_c, verbose = svm_verbose, class_weight = train_class_weight)
            cur_svm.fit(train_data_normalized, train_label_3)

            cur_searh_result_package    = resultsAnalysisSVM.main(cur_svm,         validate_combo,          validate_data_normalized, 
                                                                  validate_label_3, validate_augment_amount, classes, 'validation')
            
            _, cur_search_con_mat, _, _ = cur_searh_result_package

            
            # ==============================================================================
            cur_search_acc      = 0
            for j in range(len(cur_search_con_mat[0])):
                cur_search_acc  = cur_search_acc + cur_search_con_mat[j][j] / sum(cur_search_con_mat[j])
            cur_search_acc      = cur_search_acc / (j + 1)
            

            if cur_search_acc >= best_search_acc:
                best_search_acc = cur_search_acc
                best_svm        = cur_svm
                best_c          = cur_c
             
        
        # ==============================================================================
        test_package = loadMFCCs(test_combo, classes, dsp_package, dataset_path, data, False, unaug_dict)
        test_data,     _,   _, test_label_3,     test_dist,     test_augment_amount     = test_package
        

        # ==============================================================================
        test_data_normalized     = np.zeros((test_data.shape))
        for i in range(num_features):
            test_data_normalized[:, i]    = (test_data[:, i]    - standard_min_list[i]) / (standard_max_list[i] - standard_min_list[i])
            test_data_normalized[:, i]    = np.clip(test_data_normalized[:, i], 0, 1)
        

        # ==============================================================================
        cur_result_package = resultsAnalysisSVMOneFile.main(best_svm, test_combo, test_data_normalized)
        cur_file_acc, cur_snippet_acc, cur_file_con_mat, cur_snippet_con_mat = cur_result_package


        # ==============================================================================
        file_results.append(cur_file_acc)
        snippet_results.append(cur_snippet_acc)
        file_con_mat    = file_con_mat    + cur_file_con_mat
        snippet_con_mat = snippet_con_mat + cur_snippet_con_mat
        file_num        = file_num        + 1
     

        # ==============================================================================
        results[test_name] = [cur_file_acc,  cur_snippet_acc,  cur_file_con_mat,  cur_snippet_con_mat]
        results["Results"] = [file_results,  snippet_results,  file_con_mat,      snippet_con_mat]

        with open(dataset_path + "/" + "one_dataset_baseline_results.pickle", "wb") as temp_file_4:
                pickle.dump(results, temp_file_4, protocol = pickle.HIGHEST_PROTOCOL)


    # ==============================================================================
    cur_possible_result = ((53 - file_con_mat[0][1]) / 53 + (655 - file_con_mat[1][0]) / 655) / 2
    print('The best results we can get is:', cur_possible_result)


# ==============================================================================
# Report the final result
print('--------------------------------')
print('file results')
#print(file_results)
print(sum(file_results) / len(file_results))
print('--------------------------------')
print('snippet results')
#print(snippet_results)
print(sum(snippet_results) / len(snippet_results))
print('--------------------------------')
print('final file results')
print(file_con_mat)
acc = 0
for k in range(len(file_con_mat[0])):
    acc = acc + file_con_mat[k][k] / sum(file_con_mat[k])
print(acc / len(classes))
print('--------------------------------')
print('final snippet results')
print(snippet_con_mat)
acc = 0
for k in range(len(snippet_con_mat[0])):
    acc = acc + snippet_con_mat[k][k] / sum(snippet_con_mat[k])
print(acc / len(classes))