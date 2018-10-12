# =============================================================================
# Import Packages
from   keras.models       import Sequential
from   keras.initializers import RandomNormal
from   keras.layers       import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, normalization
from   keras              import regularizers
from   keras.callbacks    import EarlyStopping, ModelCheckpoint
from   keras.optimizers   import SGD, Adam
from   keras.losses       import binary_crossentropy
from   sklearn.utils      import class_weight
from   keras.applications import vgg16
from   keras.models       import Model
import numpy              as     np
import h5py
# =============================================================================
def CNN(train_data, train_label, validate_data, validate_label, epoch_limit, batch_size, input_shape, monitor):
    
    img_input = Input(shape = (128, 63, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    model = Model(input = img_input, output = x)

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    
    layer_names = [layer.name for layer in model.layers]

    weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    
    model.load_weights(weights_path, by_name = True)

    
    for layer_name in layer_dict.keys():
        index   = layer_names.index(layer_name)
        layer   = layer_dict[layer_name]
        print(layer)
        weights = layer.get_weights()
        model.layers[index].set_weights(weights)
    

    # pretrained_model = vgg16.VGG16(include_top  = False,
    #                            weights      = "imagenet",
    #                            input_shape  = (128, 63, 3),
    #                            )
    # for layer in pretrained_model.layers:
    #     layer.trainable = True
    # #print(pretrained_model.summary())
    output = model.output
    output = Flatten()(output)
    output = Dense(1024, activation = 'relu')(output)
    output = Dense(1024, activation = 'relu')(output)
    output = Dense(1,    activation = 'sigmoid')(output)

    
    CNN = Model(inputs = model.input, outputs = output)
    CNN.compile(loss      = binary_crossentropy,
                    optimizer = Adam(lr = 0.000001, beta_1 = 0.9, beta_2 = 0.999),
                    metrics   = ['acc'])

    # =============================================================================
    train_class_weight_raw = class_weight.compute_class_weight('balanced', np.unique(train_label), train_label)
    train_class_weight     = {}
    for a_class in np.unique(train_label):
        train_class_weight[a_class] = train_class_weight_raw[np.unique(train_label).tolist().index(a_class)]


    # =============================================================================
    early_stopping = EarlyStopping(monitor = monitor, patience = 3, verbose = 0,  mode = 'min', min_delta = 0.0001)


    # =============================================================================
    saved_path       = "best_model_this_fold.hdf5"
    model_checkpoint = ModelCheckpoint(saved_path, monitor = monitor, verbose = 0, save_best_only = True, mode = 'min')


    # =============================================================================
    history = CNN.fit(train_data,     train_label,
                    batch_size      = batch_size,
                    epochs          = epoch_limit,
                    callbacks       = [early_stopping, model_checkpoint],
                    class_weight    = train_class_weight,
                    validation_data = (validate_data, validate_label),
                    verbose         = 1,
                    shuffle         = True)


    # =============================================================================
    return CNN, history
