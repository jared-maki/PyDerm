import tensorflow as tf
from tensorflow.keras.models import Sequential  #pylint: disable=import-error
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D   #pylint: disable=import-error
from tensorflow.keras.callbacks import TensorBoard #pylint: disable=import-error
from tensorflow.keras.utils import to_categorical   #pylint: disable=import-error
import pickle
import time

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

y = to_categorical(y)

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME="{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            print(NAME)

            model = Sequential()

            model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(64, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            
            model.add(Flatten())    # Convert to 
            for l in range(dense_layer):
                model.add(Dense(dense_layer))
                model.add(Activation("relu"))

            model.add(Dense(7))
            model.add(Activation('sigmoid'))

            model.compile(loss="categorical_crossentropy",
                        optimizer="adam",
                        metrics=['accuracy'])
            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])