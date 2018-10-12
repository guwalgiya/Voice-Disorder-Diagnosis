# =============================================================================
# Import Packages
from keras              import regularizers
from keras.layers       import Input, Dense, Dropout
from keras.models       import Model
from keras.callbacks    import EarlyStopping, ModelCheckpoint
from keras.optimizers   import Adam
from keras.initializers import RandomNormal


# =============================================================================
def main(input_vector_length, x_train, x_validate, arch_bundle, train_bundle):
    

    # =============================================================================
    input_mel_spectrogram                             = Input(shape = (input_vector_length, ))
    encoder_layer, encoding_dimension, decoder_layer  = arch_bundle
    epoch_limit, batch, shuffle_choice, loss_function = train_bundle
    

    # =============================================================================
    # Encoder    
    middle_encoded = input_mel_spectrogram
    for i in range(encoder_layer):
        output_length  = encoding_dimension * (2 ** (encoder_layer - i))
        middle_encoded = Dense(output_length, kernel_initializer = 'he_normal', activation = 'relu')(middle_encoded)
    encoded = Dense(encoding_dimension, kernel_initializer = 'he_normal', activation = 'relu')(middle_encoded)
    

    # =============================================================================
    # Decoded    
    middle_decoded = encoded
    for i in range(encoder_layer):
        output_length  = encoding_dimension * (2 ** (i + 1))
        middle_decoded = Dense(output_length, kernel_initializer = 'he_normal', activation = 'relu')(middle_decoded)    
    decoded = Dense(input_vector_length, kernel_initializer = 'he_normal', activation = 'relu')(middle_decoded)
 
    
    # =============================================================================
    autoencoder = Model(inputs = input_mel_spectrogram, outputs = decoded)
    encoder     = Model(inputs = input_mel_spectrogram, outputs = encoded)


    # =============================================================================
    autoencoder.compile(optimizer = Adam(lr = 0.00001, beta_1 = 0.9, beta_2 = 0.999), loss = loss_function)
    #print(autoencoder.summary())

    # =============================================================================
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20, verbose = 0, min_delta = 0.0001, mode = 'min')
    

    # =============================================================================
    saved_path       = "best_model_this_fold.hdf5"
    model_checkpoint = ModelCheckpoint(saved_path, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'min')

    
    # =============================================================================
    history = autoencoder.fit(x_train,          x_train,
                              epochs          = epoch_limit,
                              batch_size      = batch,
                              callbacks       = [early_stopping, model_checkpoint],
                              shuffle         = shuffle_choice,
                              validation_data = (x_validate, x_validate),
                              verbose         = 0)


    # =============================================================================
    layer_size        = []
    for layer in autoencoder.layers:
        layer_size.append(layer.get_output_at(0).get_shape().as_list()[1])
    encodeLayer_index = layer_size.index(min(layer_size))

    
    # =============================================================================
    return encoder, history, encodeLayer_index
