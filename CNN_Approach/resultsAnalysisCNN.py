# =============================================================================
# Import Pacnpages
from sklearn.metrics import confusion_matrix
import numpy as np
# =============================================================================
def resultsAnalysis(myModel, test_combo, test_data, test_label_1, test_augment_amount, classes):
    
    
    # ==============================================================================
    theta             = 0.5
    predicted_label_1 = []
    predicted_label_3 = []
    predicted_label   = myModel.predict(test_data)
    for i in range(len(predicted_label)):
        if predicted_label[i][0] > predicted_label[i][1]:
            predicted_label_1.append(0)
            predicted_label_3.append("Normal")
        else:
            predicted_label_1.append(1)
            predicted_label_3.append("Pathol")
          
            
    # ==============================================================================
    snippet_index    = 0
    voted_file_label = []
    true_file_label  = []
    for i in range(len(test_combo)):
    

        # ==============================================================================
        cur_name   = test_combo[i][0]
        cur_label  = test_combo[i][1]
        cur_amount = test_augment_amount[i]

        
        # ==============================================================================
        if cur_amount == 0:
            pass
        else:         
            pathol_weight      = predicted_label_3[snippet_index : snippet_index + cur_amount].count("Pathol")
            cur_classes_weight = [cur_amount - pathol_weight, pathol_weight]
            
            
            # ==============================================================================
            if cur_classes_weight[0] > cur_classes_weight[1]:
                max_weight_index = 0
            else:
                max_weight_index = 1
            

            # ==============================================================================
            #if (cur_label == "Pathol" and max_weight_index == 0) or (cur_label == "Normal" and max_weight_index == 1):
            #if cur_label == "Normal" and max_weight_index == 1:
                #print('----------------------------')
                #print(cur_name, cur_label, cur_classes_weight)
                #for j in range(cur_amount):
                    #print(j + 1, predicted_label[snippet_index + j])

            # ==============================================================================
            # Loop Update
            snippet_index = snippet_index  + cur_amount
            voted_file_label.append(classes[max_weight_index])
            true_file_label.append(cur_label)


    # ==============================================================================
    file_acc     = 0
    file_con_mat = confusion_matrix(true_file_label, voted_file_label)
    for i in range(len(file_con_mat[0])):
        file_acc = file_acc + file_con_mat[i][i] / sum(file_con_mat[i])


    # ==============================================================================
    snippet_acc      = 0
    snippet_con_mat  = confusion_matrix(test_label_1, predicted_label_1)
    for i in range(len(snippet_con_mat[0])):
        snippet_acc  = snippet_acc + snippet_con_mat[i][i] / sum(snippet_con_mat[i])


    # ==============================================================================
    file_acc    = file_acc    / len(classes)
    snippet_acc = snippet_acc / len(classes)

    
    return [file_acc, snippet_acc, file_con_mat, snippet_con_mat]
            # for j in range(len(classes)):
            #     weight = sum(round(x[j]) for x in predicted_snippet_label[index : index + amount])
            #     #weight  = sum(x[j] for x in predicted_snippet_label[index : index + amount]) / amount
            #     cur_classes_weight.append(weight)                
            #for x in predicted_snippet_label[index : index + amount]:
                    #print(x[0])


            #max_weight_index = 0 if cur_classes_weight[0] > cur_classes_weight[1] else 1