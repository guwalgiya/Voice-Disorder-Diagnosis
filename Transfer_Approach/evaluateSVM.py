# ===============================================
# Import Packages and Functions
from sklearn.metrics import confusion_matrix
from numpy           import argmax


# ===============================================
# Main Function 1, label should be type "3"
def evaluateSVM(trained_model, test_combo, test_data, test_snippet_labels, test_augment_amount, classes):
    

    # ===============================================
    # Initialization
    snippet_index         = 0
    test_file_labels      = []
    predicted_file_labels = []


    # ===============================================
    # Get predicted results for every snippet
    predicted_snippet_labels = trained_model.predict(test_data)


    # ===============================================
    # Now anlayze each file (not snippet)
    for i in range(len(test_combo)):
        
        
        # ===============================================
        # Basic Information
        cur_file_label     = test_combo[i][1]
        cur_snippet_amount = test_augment_amount[i]


        # ===============================================
        # It might be possible that a file is too short, then it does not have any corresponding snippets
        if cur_snippet_amount == 0:
            pass
        
        
        # ===============================================
        else:
            
            
            # ===============================================
            # Now use snippets to vote for a predicted label for their file
            classes_weight   = []
            max_weight_index = 0


            # ===============================================
            for j in range(len(classes)):
                weight =  predicted_snippet_labels[snippet_index : snippet_index + cur_snippet_amount].tolist().count(classes[j])
                classes_weight.append(weight)
            

            # ===============================================
            max_weight_index = argmax(classes_weight)

            
            # ===============================================
            snippet_index  = snippet_index + cur_snippet_amount
            

            # ===============================================
            predicted_file_labels.append(classes[max_weight_index])
            test_file_labels.append(cur_file_label)




    # ===============================================
    # Find confusion matrix on filelevel and snippet level
    file_con_mat    = confusion_matrix(test_file_labels,    predicted_file_labels)
    snippet_con_mat = confusion_matrix(test_snippet_labels, predicted_snippet_labels)
    

    # ==============================================================================
    # Compute accuracy on file level and snippet level
    file_acc    = 0
    snippet_acc = 0;
    

    # ===============================================
    # file level
    for i in range(len(file_con_mat[0])):
        file_acc = file_acc + file_con_mat[i][i] / sum(file_con_mat[i])


    # ===============================================
    # snippet level
    for i in range(len(snippet_con_mat[0])):
        snippet_acc = snippet_acc + snippet_con_mat[i][i] / sum(snippet_con_mat[i])
    

    # ===============================================
    # Take average by number of classes
    file_acc    = file_acc    / len(classes)
    snippet_acc = snippet_acc / len(classes)


    # ===============================================
    # Return a list
    return [file_acc, file_con_mat, snippet_acc, snippet_con_mat]



