import numpy as np
import matplotlib.pyplot as plt
from data_adjustment import extract_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image


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
        self.categories = len(np.unique(labels))
        self.eta = eta

        self.model = Sequential()


    def add_layer(self):
        self.model.add(Conv2D(self.nfilt, (self.recf,self.recf),
                              input_shape=self.inp))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(self.nneur, activation="relu"))
        self.model.add(Dense(self.categories, activation="softmax"))

        sgd = optimizers.SGD(lr=self.eta)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd, metrics=['accuracy'])









if __name__ == '__main__':

    paths = ["/data_images/Apple",
             "/data_images/Banana"]
    true_labels = ["apple", "banana"]


    data = extract_data(paths,true_labels)
    im_shape = 20
    data.reshape(im_shape)            # making all data the same shape
    data.shuffle()
    #data.gray(); data.data = data.data[...,np.newaxis]


    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.hot_vector,
                                                        train_size=0.8)

    data_size = data.data[0].shape
    rec_field = 3
    filters = 20
    neuros_con = 50

    eta = 0.0001
    epochs = 10
    batch_size = 10

    CNN = CNN_keras(input_shape=data_size,
                    receptive_field=rec_field,
                    n_filters = filters,
                    n_neurons_connected = neuros_con,
                    labels = true_labels,
                    eta = eta)
    CNN.add_layer()


    CNN.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    print()
    scores = CNN.model.evaluate(X_test, y_test, verbose=1)




    # predicting on a image that have not been used in training or testing
    """
    test_im = np.array(Image.open("./test_images/test_apple.png"))
    test = extract_data(["/test_images"], ["apple"])
    test.reshape(20)
    print(data.data[-1:,...].shape)
    print(test.data.shape)
    """

    """
    pred = CNN.model.predict_step(test_im)
    plt.figure(figsize=[12,6])
    plt.subplot(121)
    plt.imshow(test_im, cmap="gray")
    plt.title("CNN predicts: %s = [%.3f, %.3f]" % (true_labels,
                                                   pred.numpy()[0,0],
                                                   pred.numpy()[0,1]))
    plt.subplot(122)
    plt.imshow(test_im, cmap="gray")
    plt.title("Real data")

    plt.tight_layout()
    plt.show()
    """














#
