# =============================================================================
# Import Packages
from   keras.models       import Sequential
from   keras.initializers import RandomNormal
from   keras.layers       import Conv2D, MaxPooling2D, Dense, Flatten, Conv2DTranspose, AveragePooling2D, Dropout, normalization
from   keras              import regularizers
from   keras.callbacks    import EarlyStopping, ModelCheckpoint
from   keras.optimizers   import SGD, Adam
from   keras.losses       import binary_crossentropy
from   sklearn.utils      import class_weight
import numpy              as     np


# =============================================================================
def main(train_data, train_label, validate_data, validate_label, epoch_limit, batch_size, input_shape, monitor):


    # =============================================================================
    CNN = Sequential()
    CNN.add(Conv2DTranspose(5,  kernel_size = (3, 3), strides = (1, 1), activation = 'relu', input_shape = input_shape))
    #CNN.add(MaxPooling2D(pool_size = (2, 2)))
    CNN.add(Conv2DTranspose(9, kernel_size = (3, 3), strides = (1, 1),  activation = 'relu'))
    CNN.add(MaxPooling2D(pool_size = (2, 2)))
    

    # =============================================================================
    CNN.add(Flatten())
    CNN.add(Dense(1024, activation = 'relu'))
    CNN.add(Dense(1024, activation = 'relu'))
    #CNN.add(Dense(1024, activation = 'relu'))
    CNN.add(Dense(32,   activation = 'relu'))
    CNN.add(Dense(1,    activation = 'sigmoid'))


    # =============================================================================
    CNN.compile(loss      = binary_crossentropy,
                optimizer = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999),
                metrics   = ['acc'])

    # =============================================================================
    train_class_weight_raw = class_weight.compute_class_weight('balanced', np.unique(train_label), train_label)
    train_class_weight     = {}
    for a_class in np.unique(train_label):
        train_class_weight[a_class] = train_class_weight_raw[np.unique(train_label).tolist().index(a_class)]

    
    # =============================================================================
    early_stopping = EarlyStopping(monitor = monitor, patience = 10, verbose = 0,  mode = 'min', min_delta = 0.001)


    # =============================================================================
    saved_path       = "best_model_this_fold.hdf5"
    model_checkpoint = ModelCheckpoint(saved_path, monitor = monitor, verbose = 0, save_best_only = True, mode = 'min')

    
    # =============================================================================
    history = CNN.fit(train_data,     train_label,
                    batch_size      = batch_size,
                    epochs          = epoch_limit,
                    callbacks       = [early_stopping, model_checkpoint],
                    #class_weight    = train_class_weight,
                    validation_data = (validate_data, validate_label),
                    verbose         = 0,
                    shuffle         = True)
    
    
    # =============================================================================
    return CNN, history