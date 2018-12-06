# ===============================================
# Import Packages and Functions
from   sklearn                 import svm
from   sklearn.utils           import class_weight
from   sklearn.metrics         import confusion_matrix
from   sklearn.model_selection import KFold
import numpy                   as     np
import resultsAnalysisSVM


# ===============================================
# Import Packages and Functions
def method1(train_data, train_label_2, validate_data, validate_label_2, test_data, test_label_2, test_combo, test_augment_amount):
    

    # ===============================================
    # Import Packages and Functions
    classes   =  ["Normal", "Pathol"]
    best_acc  =  0
    c_values  =  [0.1,  1,   10, 100, 1000]
    

    # ===============================================
    # Import Packages and Functions
    train_class_weight_raw = class_weight.compute_class_weight('balanced', np.unique(train_label_2), train_label_2)
    train_class_weight     = {}
    for a_class in np.unique(train_label_2):
        train_class_weight[a_class] = train_class_weight_raw[np.unique(train_label_2).tolist().index(a_class)]
    

    # ===============================================
    # Import Packages and Functions
    for c in c_values:
        cur_svm         = svm.LinearSVC(C = c, verbose = 0, tol = 0.001, max_iter = 1000, class_weight = train_class_weight)
        cur_svm.fit(train_data, train_label_2)


        # ===============================================
        # Import Packages and Functions
        vali_predict   = cur_svm.predict(validate_data)
        con_mat        = confusion_matrix(validate_label_2, vali_predict)


        # ===============================================
        # Import Packages and Functions
        combo_acc      = 0
        for i in range(len(con_mat[0])):
            combo_acc  = combo_acc + con_mat[i][i] / sum(con_mat[i])
        combo_acc      = combo_acc / (i + 1)


        # ===============================================
        # Import Packages and Functions           
        if  best_acc   < combo_acc:
            best_acc   = combo_acc;
            best_svm   = cur_svm;    
    

    # ===============================================
    # Import Packages and Functions
    result_package  = resultsAnalysisSVM.main(best_svm, test_combo, test_data, test_label_2, test_augment_amount, classes)
    

    # ===============================================
    return result_package
