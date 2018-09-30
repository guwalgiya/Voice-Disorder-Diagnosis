from sklearn                 import svm
from sklearn.metrics         import confusion_matrix
from sklearn.model_selection import KFold
import numpy                 as     np
import resultsAnalysisSVM
from   sklearn.utils           import class_weight

def method1(train_data, train_label2, validate_data, validate_label2, test_data, test_label2, test_combo, test_augment_amount):
    classes   =  ["Normal", "Pathol"]
    best_acc  =  0
    c_values  =  [0.1,  1,   10, 100]

    train_class_weight_raw = class_weight.compute_class_weight('balanced', np.unique(train_label2), train_label2)
    train_class_weight     = {}
    for a_class in np.unique(train_label2):
        train_class_weight[a_class] = train_class_weight_raw[np.unique(train_label2).tolist().index(a_class)]
    

    
    for c in c_values:
        #, class_weight = train_class_weight
        cur_svm         = svm.LinearSVC(C = c, verbose = 0)# class_weight = train_class_weight ) 
        cur_svm.fit(train_data, train_label2)

        vali_predict   = cur_svm.predict(validate_data)
        con_mat        = confusion_matrix(validate_label2, vali_predict)

        combo_acc      = 0
        for i in range(len(con_mat[0])):
            combo_acc  = combo_acc + con_mat[i][i] / sum(con_mat[i])
        combo_acc      = combo_acc / (i + 1)
            
        if  best_acc   < combo_acc:
            best_acc   = combo_acc;
            best_svm   = cur_svm;    

    result_package  = resultsAnalysisSVM.main(best_svm, test_combo, test_data, test_label2, test_augment_amount, classes)

    return result_package
