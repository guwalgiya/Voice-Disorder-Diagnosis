import numpy           as     np
from   sklearn         import svm
from   sklearn.metrics import confusion_matrix

def main(train_data, train_dist, vali_data, vali_label, test_data, test_label):
    gamma_values = [0.01, 0.1, 1, 10]
  
    #classifier = svm.OneClassSVM(kernel = "rbf", gamma = )   
    for g in gamma_values:
        best_acc       = 0
        outliers_fraction = train_dist["Pathol"][1] / (train_dist["Normal"][1] + train_dist["Pathol"][1])
        classifier     = svm.OneClassSVM(nu = 0.95 * outliers_fraction + 0.05, kernel = "rbf", gamma = g)
        classifier.fit(train_data)

        vali_predict   = classifier.predict(vali_data)
        #con_mat        = confusion_matrix(vali_label, vali_predict)
        
        predict = []
        for p in vali_predict:
            if p == -1:
                predict.append("Pathol")
            else:
                predict.append("Normal")
        #print(vali_predict[1])
        #print(vali_label[1])
        
        con_mat        = confusion_matrix(vali_label, predict)
        combo_acc      = 0
        for i in range(len(con_mat[0])):
            combo_acc  = combo_acc + con_mat[i][i] / sum(con_mat[i])
        combo_acc      = combo_acc / (i + 1)
            
        if  best_acc   < combo_acc:
            best_acc   = combo_acc;
            best_svm   = classifier;
            #print('We have a new best acc')

    test_predict = best_svm.predict(test_data)
    predict  = []
    for p in test_predict:
        if p == -1:
            predict.append("Pathol")
        else:
            predict.append("Normal")
    con_mat      = confusion_matrix(test_label, predict)     
    print(con_mat)
    test_acc      = 0
    for i in range(len(con_mat[0])):
        test_acc  = test_acc + con_mat[i][i] / sum(con_mat[i])
        test_acc      = test_acc / (i + 1)
    #print(test_acc)

    return test_acc
