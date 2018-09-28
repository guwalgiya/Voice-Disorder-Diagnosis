from   keras.models            import Sequential
from   keras.layers            import Conv2D, MaxPooling2D, Dense, Flatten
from   sklearn.model_selection import KFold
from   sklearn.metrics         import confusion_matrix
import numpy                   as     np
import keras
import math
import getCombination
import dataSplit
import loadMelSpectrogram
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# Dataset Initialization
dataset_main_path_train = "/home/hguan/7100-Master-Project/Dataset-Spanish"
dataset_main_path_test  = "/home/hguan/7100-Master-Project/Dataset-KeyPentax"
classes = ['Normal','Pathol']
train_percent      = 90
validate_percent   = 10
test_percent       = 0

# =============================================================================
fs                  = 16000
snippet_length      = 500  #in milliseconds
snippet_hop         = 100 #in ms
mel_length          = 128
block_size          = 512
hop_size            = 256
package             = [snippet_length, snippet_hop, block_size, hop_size, mel_length]
input_vector_length = mel_length * math.ceil(snippet_length / 1000 * fs / hop_size)
input_shape         = (mel_length, math.ceil(snippet_length / 1000 * fs / hop_size),1)

# =============================================================================
name_class_combo_train = getCombination.main(dataset_main_path_train, classes)
name_class_combo_test  = getCombination.main(dataset_main_path_test,  classes)

# =============================================================================
[train_combo, validate_combo,          _]  = dataSplit.main(name_class_combo_train, train_percent, validate_percent, test_percent)
[_,           _,              test_combo]  = dataSplit.main(name_class_combo_test,  0,             0,                100)
print(len(train_combo), len(validate_combo), len(test_combo))
# =============================================================================
train_package     = loadMelSpectrogram.main(train_combo,    'training',    classes, package, fs, dataset_main_path_train, input_vector_length)   
validate_package  = loadMelSpectrogram.main(validate_combo, 'validating',  classes, package, fs, dataset_main_path_train, input_vector_length)   
test_package      = loadMelSpectrogram.main(test_combo,     'testing',     classes, package, fs, dataset_main_path_test,  input_vector_length)
print(train_package)
train_data,    train_label,    _,           train_dist,    _                   = train_package
validate_data, validate_label, _,           validate_dist, _                   = validate_package
test_data,     test_label,     test_label2, test_dist,     test_augment_amount = test_package
print(len(train_data))
print(len(validate_data))
print(len(test_data))
myModel = Sequential()
layer = Conv2D(16, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', input_shape = input_shape)
myModel.add(layer)
pool = MaxPooling2D(pool_size = (2, 2),  strides = (2, 2))
myModel.add(pool)
myModel.add(Conv2D(8, (3, 3), activation='relu'))
myModel.add(MaxPooling2D(pool_size=(2, 2)))   

myModel.add(Flatten())
myModel.add(Dense(1024, activation = 'relu'))
myModel.add(Dense(256, activation = 'relu'))
myModel.add(Dense(64, activation = 'relu'))
myModel.add(Dense(len(classes), activation = 'softmax'))

myModel.compile(loss      = keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.SGD(lr = 0.001),
                metrics   = ['accuracy'])

myModel.fit(train_data, train_label,
            batch_size      = 256,
            epochs          = 100,
            verbose         = 0,
            validation_data = (validate_data, validate_label))

file_acc, snippet_acc, file_con_mat, snippet_con_mat = resultsAnalysis.main(myModel, test_combo, test_data, test_label2, test_augment_amount, classes)

print('--------------------------------')
print('file results')
print(file_acc)
print('--------------------------------')
print('snippet results')
print(snippet_acc)
print('--------------------------------')
print('final file results')
print(file_con_mat)
acc = 0;
for i in range(len(file_con_mat[0])):
    acc = acc + file_con_mat[i][i] / sum(file_con_mat[i])
print(acc / 2)
print('--------------------------------')
print('final snippet results')
print(snippet_con_mat)
acc = 0;
for i in range(len(snippet_con_mat[0])):
    acc = acc + snippet_con_mat[i][i] / sum(snippet_con_mat[i])
print(acc / 2)