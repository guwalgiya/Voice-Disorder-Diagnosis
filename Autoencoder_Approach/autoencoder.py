# ===============================================
# Import Packages and Functions
from keras.layers       import Input, Dense
from keras.models       import Model
from keras.callbacks    import EarlyStopping, ModelCheckpoint
from keras.optimizers   import Adam


# ===============================================
# Main Function
def myAutoencoder(train_data, validation_data, AE_architecture_package, AE_training_package, AE_callbacks_package):
    
    
    # ===============================================
    # Load Parameters
    input_vector_length, encoding_dimension, num_encoding_layer, num_decoding_layer                                                                             = AE_architecture_package
    saved_model_name,    callbacks_mode,     callbacks_monitor,  callbacks_patience, callbacks_min_delta, callbacks_verbose, if_only_save_best                  = AE_callbacks_package
    learning_rate,       epoch_limit,        batch_size,         shuffle_choice,     loss_function,       adam_beta_1,       adam_beta_2,      training_verbose = AE_training_package

    
    # ===============================================
    # Autoencoder's first layer
    input_layer = Input(shape = (input_vector_length, ))
    

    # ===============================================
    # Encoder Part 1
    middle_encoded = input_layer
    for i in range(num_encoding_layer):
        output_length  = encoding_dimension * (2 ** (num_encoding_layer - i))
        middle_encoded = Dense(output_length, kernel_initializer = "he_normal", activation = "relu")(middle_encoded)


    # ===============================================
    # Encoder Part 2: encoded layer at the middle
    encoded = Dense(encoding_dimension, kernel_initializer = "he_normal", activation = "relu")(middle_encoded)
    

    # ===============================================
    # Decoder Part 1  
    middle_decoded = encoded
    for i in range(num_encoding_layer):
        output_length  = encoding_dimension * (2 ** (i + 1))
        middle_decoded = Dense(output_length, kernel_initializer = "he_normal", activation = "relu")(middle_decoded)  


    # ===============================================
    # Decoder Part 2: the last layer 
    decoded = Dense(input_vector_length, kernel_initializer = "he_normal", activation = "relu")(middle_decoded)
 
    
    # ===============================================
    # Define the whole autoencoder and encoder (encoder is a part of autoencoder)
    autoencoder = Model(inputs = input_layer, outputs = decoded)
    encoder     = Model(inputs = input_layer, outputs = encoded)

    
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
                                       save_best_only    = if_only_save_best)

    
    # ===============================================
    autoencoder.compile(loss      = loss_function,
                        optimizer = Adam(lr     = learning_rate, 
                                         beta_1 = adam_beta_1, 
                                         beta_2 = adam_beta_2))



    # ===============================================
    # when training, the two end-points of an AE should be the same
    training_history = autoencoder.fit(train_data,       
                                       train_data,
                                       epochs          = epoch_limit,
                                       verbose         = training_verbose,
                                       shuffle         = shuffle_choice,
                                       callbacks       = [early_stopping, model_checkpoint],
                                       batch_size      = batch_size,
                                       validation_data = (validation_data, validation_data))


    # ===============================================
    # We want to find the index of the encoding layer
    layer_sizes = []
    for a_layer in autoencoder.layers:
        layer_sizes.append(a_layer.get_output_at(0).get_shape().as_list()[1])
    

    # ===============================================
    # Get the encoding layer's index
    encoding_layer_index = layer_sizes.index(min(layer_sizes))

    
    # ===============================================
    return training_history, encoding_layer_index
