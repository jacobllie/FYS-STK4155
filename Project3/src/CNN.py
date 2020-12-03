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
             "/data_images/Banana",
             "/data_images/Kiwi",
             "/data_images/Tomato"]
    true_labels = ["apple", "banana", "kiwi", "tomato"]




    data = extract_data(paths, true_labels, lim_data=750)
    im_shape = 50
    data.reshape(im_shape)            # making all data the same shape
    data.shuffle()
    #data.gray()

    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.hot_vector,
                                                        train_size=0.8)

    data_size = data.data[0].shape
    rec_field = 3
    filters = 20
    neuros_con = 50

    eta = 0.0001
    epochs = 5
    batch_size = 20

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

    #test_im = np.array(Image.open("./test_images/test_apple.png"))
    test = extract_data(["/test_images"], ["apple"])
    test.reshape(im_shape)
    #test.gray()

    pred = CNN.model.predict_step(test.data[:1,...])
    plt.figure(figsize=[12,6])
    plt.subplot(121)
    plt.imshow(test.data[-1,...], cmap="gray")
    plt.title("CNN predicts: %s" % (true_labels[np.argmax(pred.numpy())]))
    for i in range(len(true_labels)):
        plt.plot([0],[0], label="%s: %.3f" % (true_labels[i], pred.numpy()[0,i]))
    plt.legend()
    plt.subplot(122)
    plt.imshow(test.real_data[-1,...], cmap="gray")
    plt.title("Real data")

    plt.tight_layout()
    plt.show()














#
