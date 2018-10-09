from   sklearn.metrics import confusion_matrix

def main(myModel, test_combo, test_data, test_label2, test_augment_amount, classes):
    index        = 0
    count        = 0
    voted_labels = []
    true_labels  = []
    prediction   = myModel.predict(test_data)
    for i in range(len(test_combo)):
        #print('--------------------')
    
        name   = test_combo[i][0]
        label  = test_combo[i][1]
        amount = test_augment_amount[i]
        if amount == 0:
            pass
        else:
            #print(name, label, test_augment_amount[i])
            count = count + amount
         
            classes_weight = []
            max_weight_index = 0
            for j in range(len(classes)):
                weight =  prediction[index : index + amount].tolist().count(classes[j])
                classes_weight.append(weight)

            max_weight_index = 0 if classes_weight[0] > classes_weight[1] else 1
            #print(classes_weight)
            #print(max_weight_index)

            if (label == "Pathol" and max_weight_index == 0) or (label == "Normal" and max_weight_index == 1):
            #if label == "Normal":
                print(name, label, classes_weight)
                
            index = index  + amount
            voted_labels.append(classes[max_weight_index])
            true_labels.append(label)
    voted_con_mat = confusion_matrix(true_labels, voted_labels)

    #print("Voted Results")
    #print(voted_con_mat)
    voted_acc = 0;
    for i in range(len(voted_con_mat[0])):
        voted_acc = voted_acc + voted_con_mat[i][i] / sum(voted_con_mat[i])
    voted_acc = voted_acc / len(classes)
    #print(voted_acc)

       





    #print("Snippets Results")
    con_mat   = confusion_matrix(test_label2, prediction)
    #print(con_mat)
    acc = 0;
    for i in range(len(con_mat[0])):
        acc = acc + con_mat[i][i] / sum(con_mat[i])
    acc = acc / len(classes)
    #print(acc)


    #print('Check')
    #print(len(test_data), count)

    return [voted_acc, voted_con_mat, acc, con_mat]
