from keras.layers import Input, Dense
from keras.models import Model

def main(input_vector_length, arch_bundle):
    
    input_mel_spectrogram                            = Input(shape = (input_vector_length, ))
    encoder_layer, encoding_dimension, decoder_layer = arch_bundle
    
    encoded = input_mel_spectrogram
    for i in range(encoder_layer):
        encoded = Dense(encoding_dimension * (2 ** (encoder_layer - 1 - i)), kernel_initializer = 'RandomUniform', bias_initializer = 'zeros')(encoded)
        
    encoder = Model(input_mel_spectrogram, encoded)

    
    return encoder
