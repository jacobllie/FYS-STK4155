import numpy as np
import matplotlib.pyplot as plt
from data_adjustment import extract_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CNN_keras:
    def __init__(self, input_shape, receptive_field, n_filters,
                                    n_neurons_connected, labels, eta, lmbd=0):
        """
        input_shape: the shape of input data
        receptive_field: small area connected to each neuron in next layer
        n_filters:
        n_neurons_connected:
        labels:
        """

        self.inp = input_shape
        self.recf = receptive_field
        self.nfilt = n_filters
        self.nneur = n_neurons_connected
        self.labels = to_categorical(labels)
        self.categories = len(np.unique(labels))
        self.eta = eta

        self.model = Sequential()


    def add_layer(self, eta):
        self.model.add(Conv2D(self.nfilt, (self.recf,self.recf),
                              input_shape=self.inp))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.nneur, activation="relu"))
        self.model.add(Dense(self.categories, activation="softmax"))

        sgd = optimizer.SGD(lr=eta)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd, metrics=['accuracy'])











if __name__ == '__main__':

    paths = ["/data_images/Apple",
             "/data_images/Banana"]
    labels = ["apple", "banana"]


    data = extract_data(paths,labels)
    data.reshape(50)            # making all data the same shape

    print(data.data[0].shape)

    data_size = data.data[0].shape
    rec_field = 3
    filters = 20
    neuros_con = 50
    categories = len(labels)
    eta = 0.01
    epochs = 10



















#
