import numpy as np

# =============================================================================
def main(myModel, test_combo, test_data):


    # =============================================================================
    name       = test_combo[0][0]
    true_label = 0 if test_combo[0][1] == "Normal" else 1
    num_true   = 0
    num_false  = 0


    # =============================================================================
    model_predictions = myModel.predict(test_data)

    for prediction in model_predictions :
        if round(prediction[0]) == true_label:
            num_true  = num_true  + 1
        else:
            num_false = num_false + 1 
    

    # =============================================================================
    snippet_acc     = num_true / len(model_predictions)

    if true_label == 0:
            snippet_con_mat = np.array([[num_true,num_false],[0,0]])
    else:
            snippet_con_mat = np.array([[0,0],[num_false,num_true]])



    # =============================================================================
    if num_true > num_false:
        file_acc  = 1
        if true_label == 0:
            file_con_mat = np.array([[1,0],[0,0]])
        else:
            file_con_mat = np.array([[0,0],[0,1]])
    else:
        file_acc = 0
        if true_label == 0:
            file_con_mat = np.array([[0,1],[0,0]])
        else:
            file_con_mat = np.array([[0,0],[1,0]])
    
    # =============================================================================
    print(name, test_combo[0][1], bool(file_acc))
    return [file_acc, snippet_acc, file_con_mat, snippet_con_mat]
    