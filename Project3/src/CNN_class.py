import numpy as np
import matplotlib.pyplot as plt
from data_adjustment import extract_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers, regularizers,initializers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import time
import sys

class CNN_keras:
    def __init__(self, input_shape, receptive_field, n_filters,
                                    n_neurons_connected, labels, eta, lmbd):
        """
        input_shape: the shape of input data
        receptive_field: small area connected to each neuron in next layer (size of filters I think!?!?!?!?!)
        n_filters: number of filters that is multiplied to the input image
        n_neurons_connected:
        labels:
        """

        self.inp = input_shape
        self.recf = receptive_field
        self.nfilt = n_filters
        self.nneur = n_neurons_connected
        self.categories = len(np.unique(labels))
        self.eta = eta
        self.lmbd = lmbd

        self.model = Sequential()


    def add_layer(self, show_model=False):
        """
        First convolutional layer must contain the input shape,
        the other layers is not dependent of it
        """
        self.model.add(Conv2D(self.nfilt, (self.recf,self.recf),
            input_shape=self.inp,activation='relu', padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(self.nfilt, (self.recf,self.recf),
            input_shape=self.inp,activation='relu', padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))


        """
        Before we can add dense layers, the output from previous
        3D layers must be flatten (i.e. convert to vector form)
        """
        self.model.add(Flatten())
        self.model.add(Dense(self.nneur, activation="relu",
                             kernel_regularizer=regularizers.l2(self.lmbd)))
        self.model.add(Dense(self.categories, activation="softmax",
                             kernel_regularizer=regularizers.l2(self.lmbd)))

        sgd = optimizers.SGD(lr=self.eta)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd, metrics=['accuracy'])

        if show_model==True: self.model.summary()
