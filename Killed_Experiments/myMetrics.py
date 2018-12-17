from   keras      import backend as K 
import tensorflow as     tf
def macroAccuracy(label_true, label_predict):

    label_predict       = K.round(label_predict)
    possible_negatives  = K.sum(K.round(K.clip(label_true, 0, 1)))
    true_negatives      = K.sum(K.round(K.clip(label_true * label_predict, 0, 1)))
    false_negatives     = possible_negatives  - true_negatives 

    predicted_negatives = K.sum(K.round(K.clip(label_predict, 0, 1)))
    false_positives     = predicted_negatives  - true_negatives 
    
    all_points          = K.sum(K.round(K.clip(label_true, 1, 1)))
    true_positives      = all_points - true_negatives  - false_positives - false_negatives 
    
    positives_acc       = true_positives / (true_positives + false_positives + K.epsilon()) # for Normal
    negatives_acc       = true_negatives / (true_negatives + false_negatives + K.epsilon()) # for Pathol
    
    # sess = tf.InteractiveSession()
    # print(positives_acc.eval())
    # print(negatives_acc.eval())
    # print(true_positives.eval())
    # print(false_positives.eval())
    # print(false_negatives.eval())
    # print(true_negatives.eval())
    return 0.5 * (positives_acc + negatives_acc) 