# ===============================================
# Import Packages and Functions
from   keras.layers       import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from   keras.models       import Sequential
from   keras.losses       import binary_crossentropy
from   sklearn.utils      import class_weight
from   keras.callbacks    import EarlyStopping, ModelCheckpoint
from   keras.optimizers   import Adam
import numpy              as     np


# ===============================================
def myVGGish(train_data, train_snippet_labels_1, train_snippet_labels_2, validate_data, validate_snippet_labels_2, classes, training_based_on, VGGish_train_shape, trainable_list, CNN_training_package, CNN_callbacks_package):
    

    # ===============================================
    # Load parameters for model's training
    learning_rate, epoch_limit, batch_size, metric, shuffle_choice, loss_function, adam_beta_1, adam_beta_2, training_verbose = CNN_training_package
    

    # ===============================================
    # Load parameters for training's callbacks (Early Stopping and Model Checkpoints)
    saved_model_name, callbacks_mode, callbacks_monitor,  callbacks_patience, callbacks_min_delta, callbacks_verbose, callbacks_if_save_best = CNN_callbacks_package
    
    train_b1, train_b2, train_b3, train_b4, train_fc = trainable_list

    # ===============================================
    VGGish = Sequential()
    

    # ===============================================
    # Block 1
    VGGish.add(Conv2D(64, (3, 3), activation = "relu", padding = "same", name = "conv1", input_shape = VGGish_train_shape, trainable = train_b1))
    VGGish.add(MaxPooling2D((2, 2), strides = (2, 2), name = "pool1", trainable = train_b1))


    # ===============================================
    # Block 2
    VGGish.add(Conv2D(128, (3, 3), activation = "relu", padding = "same", name = "conv2", trainable = train_b2))
    VGGish.add(MaxPooling2D((2, 2), strides = (2, 2), name = "pool2", trainable = train_b2))
    

    # ===============================================
    # Block 3
    VGGish.add(Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "conv3_1", trainable = train_b3))
    VGGish.add(Conv2D(256, (3, 3), activation = "relu", padding = "same", name = "conv3_2", trainable = train_b3))
    VGGish.add(MaxPooling2D((2, 2), strides = (2, 2), name = "pool3", trainable = train_b3))
    

    # ===============================================
    # Block 4
    VGGish.add(Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "conv4_1", trainable = train_b4))
    VGGish.add(Conv2D(512, (3, 3), activation = "relu", padding = "same", name = "conv4_2", trainable = train_b4))
    VGGish.add(MaxPooling2D((2, 2), strides = (2, 2), name = "pool4", trainable = train_b1))
    

    # ===============================================
    # Block fc
    VGGish.add(Flatten(name = "flatten"))
    VGGish.add(Dense(4096, activation = "relu", name = "fc1_1", trainable = train_fc))
    VGGish.add(Dropout(0.5))
    VGGish.add(Dense(4096, activation = "relu", name = "fc1_2", trainable = train_fc))
    VGGish.add(Dropout(0.5))
    VGGish.add(Dense(128,  activation = "relu", name = "fc2",   trainable = train_fc))


    # ===============================================
    if training_based_on == "VGGish":
        VGGish.load_weights("vggish_weights.ckpt")
        VGGish.add(Dense(64,           activation = "relu"))
        # VGGish.add(Dense(64,           activation = "relu"))
        # VGGish.add(Dense(32,           activation = "relu"))
        # VGGish.add(Dense(32,           activation = "relu"))
        # VGGish.add(Dense(16,           activation = "relu"))
        # VGGish.add(Dense(16,           activation = "relu"))
        VGGish.add(Dense(len(classes), activation = "softmax"))
    else:
        VGGish.add(Dense(64,           activation = "relu"))
        # VGGish.add(Dense(64,           activation = "relu"))
        # VGGish.add(Dense(32,           activation = "relu"))
        # VGGish.add(Dense(32,           activation = "relu"))
        # VGGish.add(Dense(16,           activation = "relu"))
        # VGGish.add(Dense(16,           activation = "relu"))
        VGGish.add(Dense(len(classes), activation = "softmax"))
        VGGish.load_weights(saved_model_name)
    
    
    # ===============================================
    # Compile CNN
    VGGish.compile(loss      = binary_crossentropy,
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
    training_history = VGGish.fit(train_data,     train_snippet_labels_2,
                                  epochs          = epoch_limit,
                                  verbose         = training_verbose,
                                  shuffle         = shuffle_choice,
                                  callbacks       = [early_stopping, model_checkpoint],
                                  batch_size      = batch_size,
                                  class_weight    = train_class_weight,
                                  validation_data = (validate_data, validate_snippet_labels_2))


    # ===============================================
    return training_history
