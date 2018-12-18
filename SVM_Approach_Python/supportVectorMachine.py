# ===============================================
# Import Packages and Functions
from   sklearn                 import svm
from   evaluateSVM             import evaluateSVM
from   sklearn.utils           import class_weight
from   sklearn.metrics         import confusion_matrix
import numpy                   as     np


# ===============================================
# Main Function, label should be type "3"
def mySVM(train_data, train_snippet_labels, validate_data, validate_snippet_labels, test_data, test_snippet_labels, test_combo, test_augment_amount, classes, svm_package):
    

    # ===============================================
    # Load Parameters
    c_values, svm_verbose, svm_tolerance, svm_max_iteration = svm_package

     
    # ===============================================
    # Find class weight step 1, because dataset might be unbalanced
    # This will return an array, example: array([0.75 1.5])
    train_class_weight_raw = class_weight.compute_class_weight("balanced", np.unique(train_snippet_labels), train_snippet_labels)
    

    # ===============================================
    # Find class weight step 2
    # This will creat a dictionary, example: {'Normal': 0.75, 'Pathol': 1.5}
    train_class_weight = {}
    for a_class in np.unique(train_snippet_labels):
        train_class_weight[a_class] = train_class_weight_raw[np.unique(train_snippet_labels).tolist().index(a_class)]
    
    
    # ===============================================
    # start to look for the best c value
    best_snippet_acc = 0
    for c in c_values:
        

        # ===============================================
        # create a linear svm
        cur_svm = svm.LinearSVC(C = c, verbose = svm_verbose, tol = svm_tolerance, max_iter = svm_max_iteration, class_weight = train_class_weight) 


        # ===============================================
        # use training set to train the svm
        cur_svm.fit(train_data, train_snippet_labels)
         

        # ===============================================
        # use validate set to check the svm
        predicted_snippet_labels = cur_svm.predict(validate_data)
        con_mat                  = confusion_matrix(validate_snippet_labels, predicted_snippet_labels)
        

        # ===============================================
        # Find macro snippet accuracy under the current parameter setting
        cur_snippet_acc      = 0
        for i in range(len(con_mat[0])):
            cur_snippet_acc  = cur_snippet_acc + con_mat[i][i] / sum(con_mat[i])
        cur_snippet_acc      = cur_snippet_acc / len(classes)
         

        # ===============================================
        # Choose the best svm parameter
        if  best_snippet_acc < cur_snippet_acc:
            best_snippet_acc = cur_snippet_acc;
            best_svm         = cur_svm;    
    

    # ===============================================
    # Evaluate the best svm after searching c-values, using test data
    result_package  = evaluateSVM(best_svm, test_combo, test_data, test_snippet_labels, test_augment_amount, classes)
    

    # ===============================================
    return result_package
