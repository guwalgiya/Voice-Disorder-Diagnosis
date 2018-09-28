# =============================================================================
# Import Packages
from   keras.models       import Sequential
from   keras.initializers import RandomNormal
from   keras.layers       import Conv2D, MaxPooling2D, Dense, Flatten, Conv2DTranspose, AveragePooling2D, Dropout
from   keras              import regularizers
from   keras.callbacks    import EarlyStopping
from   keras.optimizers   import SGD
from   myMetrics          import macroAccuracy
from   sklearn.utils      import class_weight
import numpy              as     np
import resultsAnalysis
import keras


# =============================================================================
def main(train_data, train_label, validate_data, validate_label, epoch_limit, batch_size, input_shape):


    # =============================================================================
    CNN = Sequential()
    CNN.add(Conv2DTranspose(9,  kernel_size = (5, 5),       strides = (1, 1), 
                            data_format = 'channels_first', activation = 'relu', 
                            input_shape = input_shape,      kernel_initializer = 'he_normal'))
    CNN.add(MaxPooling2D(pool_size = (2, 2)))
    CNN.add(Conv2DTranspose(21, kernel_size = (3, 3),       strides = (1, 1), 
                            data_format = 'channels_first', activation = 'relu', kernel_initializer = 'he_normal'))
    CNN.add(MaxPooling2D(pool_size = (2, 2)))
    

    # =============================================================================
    CNN.add(Flatten())
    CNN.add(Dense(4096, activation = 'relu',    kernel_initializer = 'he_normal'))
    CNN.add(Dropout(0.2))
    CNN.add(Dense(1024, activation = 'relu',    kernel_initializer = 'he_normal'))
    CNN.add(Dropout(0.2))
    CNN.add(Dense(64,   activation = 'relu',    kernel_initializer = 'he_normal'))
    CNN.add(Dense(1,    activation = 'sigmoid', kernel_initializer = 'he_normal'))


    # =============================================================================
    CNN.compile(loss      = keras.losses.binary_crossentropy,
                optimizer = SGD(lr = 0.0005, momentum = 0.90, decay = 0.0001, nesterov = False),
                metrics   = [macroAccuracy])
    

    # =============================================================================
    early_stopping = EarlyStopping(monitor = 'val_macroAccuracy', patience = 2, verbose = 0,  mode = 'max', min_delta = 0.001)
    

    # =============================================================================
    train_class_weight_raw = class_weight.compute_class_weight('balanced', np.unique(train_label), train_label)
    train_class_weight     = {}
    for a_class in np.unique(train_label):
        train_class_weight[a_class] = train_class_weight_raw[np.unique(train_label).tolist().index(a_class)]
    

    # =============================================================================
    CNN.fit(train_data,     train_label,
                batch_size      = batch_size,
                epochs          = epoch_limit,
                callbacks       = [early_stopping],
                class_weight    = train_class_weight,
                validation_data = (validate_data, validate_label),
                verbose         = 0,
                shuffle         = True)
    

    # =============================================================================
    return CNN