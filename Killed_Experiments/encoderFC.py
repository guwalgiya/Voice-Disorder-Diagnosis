from keras.layers     import Dense
from keras.models     import Sequential
from keras.optimizers import SGD
def main(train, train_label, validate, validate_label, test, test_label, encoder, num_classes, train_bundle):
    [epochs, batch] = train_bundle
    model = Sequential()
    model.add(encoder)
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(loss = 'mean_squared_error',
		  optimizer = SGD(lr = 0.01),
		  metrics = ['accuracy'])
    model.fit(train, train_label, 
              batch_size = batch,
              epochs = epochs,
              verbose = 1,
              validation_data = (validate, validate_label))
    score = model.evaluate(test, test_label, verbose = 0)
    print('Test Acccuracy: ', score[1])
    return score[1]
