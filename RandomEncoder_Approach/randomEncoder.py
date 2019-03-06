# ===============================================
# Import Packages and Functions
from keras.layers       import Input, Dense
from keras.models       import Model


# ===============================================
# Main Function
def myEncoder(RE_architecture_package):
    
    
    # ===============================================
    # Load Parameters
    input_vector_length, encoding_dimension, num_encoding_layer, num_decoding_layer = RE_architecture_package     
    

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
    # Define the whole autoencoder and encoder (encoder is a part of autoencoder)
    encoder = Model(inputs = input_layer, outputs = encoded)


    # ===============================================
    # We want to find the index of the encoding layer
    layer_sizes = []
    for a_layer in encoder.layers:
        layer_sizes.append(a_layer.get_output_at(0).get_shape().as_list()[1])
    

    # ===============================================
    # Get the encoding layer's index
    encoding_layer_index = layer_sizes.index(min(layer_sizes))

    
    # ===============================================
    return encoder, encoding_layer_index
