# ===============================================
# Import Packages and Functions
from   keras.models       import Sequential
from   keras.layers       import AveragePooling2D, Dense, Flatten, Conv2DTranspose
from   keras.losses       import binary_crossentropy
from   sklearn.utils      import class_weight
from   keras.callbacks    import EarlyStopping, ModelCheckpoint
from   keras.optimizers   import Adam
import numpy              as     np


# ===============================================
# Main Function. Label type 1: Normal = 0, Pathol = 1, Label type 2: Normal = [1, 0], Pathol = [0, 1]
def myCNN(train_data, train_snippet_labels_1, train_snippet_labels_2, validate_data, validate_snippet_labels_2, classes, CNN_architecture_package, CNN_training_package, CNN_callbacks_package):


    # ===============================================
    # Load parameters for the model's architecture
    input_shape, FC_num_neuron_list = CNN_architecture_package


    # ===============================================
    # Load parameters for model's training
    learning_rate, epoch_limit, batch_size, metric, shuffle_choice, loss_function, adam_beta_1, adam_beta_2, training_verbose = CNN_training_package
    

    # ===============================================
    # Load parameters for training's callbacks (Early Stopping and Model Checkpoints)
    saved_model_name, callbacks_mode, callbacks_monitor,  callbacks_patience, callbacks_min_delta, callbacks_verbose, callbacks_if_save_best = CNN_callbacks_package
    

    # ===============================================
    # Initialize
    CNN = Sequential()


    # ===============================================
    # Add Convolution Layers
    CNN.add(Conv2DTranspose(9,  kernel_size = (5, 5), strides = (1, 1), activation = "relu", input_shape = input_shape))
    CNN.add(Conv2DTranspose(15, kernel_size = (3, 3), strides = (1, 1), activation = "relu"))


    # ===============================================
    # Add a Pooling Layer
    CNN.add(AveragePooling2D(pool_size = (2, 2)))
    

    # ===============================================
    # Fully Connected Layers (not the last softmax layer)
    CNN.add(Flatten())
    for num_neuron in FC_num_neuron_list:
        CNN.add(Dense(num_neuron, activation = "relu"))
    
    
    # ===============================================
    # Softmax Layer at the end
    CNN.add(Dense(len(classes), activation = "softmax"))


    # ===============================================
    # Compile CNN
    CNN.compile(loss      = binary_crossentropy,
                metrics   = [metric],
                optimizer = Adam(lr     = learning_rate, 
                	             beta_1 = adam_beta_1, 
                	             beta_2 = adam_beta_2))


    # ===============================================
    # Find class weight step 1, because dataset might be unbalanced
    # This will return an array, example: array([0.75 1.5])
    train_class_weight_raw = class_weight.compute_class_weight("balanced", np.unique(train_snippet_labels_1), train_snippet_labels_1)
    

    # ===============================================
    # Find class weight step 2
    # This will creat a dictionary, example: {'Normal': 0.75, 'Pathol': 1.5}
    train_class_weight              = {}
    for a_label in np.unique(train_snippet_labels_1):
        train_class_weight[a_label] = train_class_weight_raw[np.unique(train_snippet_labels_1).tolist().index(a_label)]

    
    # ===============================================
    # Define early stopping
    early_stopping = EarlyStopping(mode      = callbacks_mode,
                                   monitor   = callbacks_monitor,
                                   verbose   = callbacks_verbose,
                                   patience  = callbacks_patience,
                                   min_delta = callbacks_min_delta)
    

    # ===============================================
    # Define model checkpoint
    model_checkpoint = ModelCheckpoint(saved_model_name, 
                                       mode              = callbacks_mode,
                                       monitor           = callbacks_monitor, 
                                       verbose           = callbacks_verbose,  
                                       save_best_only    = callbacks_if_save_best)

  
    # ===============================================
    # Training
    training_history = CNN.fit(train_data,     
                               train_snippet_labels_2,
                               epochs                 = epoch_limit,
                               verbose                = training_verbose,
                               shuffle                = shuffle_choice,
                               callbacks              = [early_stopping, model_checkpoint],
                               batch_size             = batch_size,
                               class_weight           = train_class_weight,
                               validation_data        = (validate_data, validate_snippet_labels_2))
                    
            
    # ===============================================
    return training_history