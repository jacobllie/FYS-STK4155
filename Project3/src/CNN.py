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
import os


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
                              input_shape=self.inp))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        #self.model.add(Conv2D(2*self.nfilt, (self.recf,self.recf)))
        #self.model.add(MaxPooling2D(pool_size=(2,2)))


        """
        Before we can add dense layers, the output from previous
        3D layers must be flatten (i.e. convert to vector form)
        """
        self.model.add(Flatten())
        #self.model.add(Dense(2*self.nneur, activation="relu",
        #                     kernel_regularizer=regularizers.l2(self.lmbd)))
        self.model.add(Dense(self.nneur, activation="relu",
                             kernel_regularizer=regularizers.l2(self.lmbd)))
        self.model.add(Dense(self.categories, activation="softmax",
                             kernel_regularizer=regularizers.l2(self.lmbd)))

        sgd = optimizers.SGD(lr=self.eta)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd, metrics=['accuracy'])

        if show_model==True: self.model.summary()







if __name__ == '__main__':

    paths = ["/data_images/Apple",
             "/data_images/Banana",
             "/data_images/Kiwi",
             "/data_images/Mango",
             "/data_images/Orange",
             "/data_images/Pear",
             "/data_images/Tomato"]
    true_labels = ["apple", "banana", "kiwi", "mango",
                   "orange", "pear", "tomato"]

    #paths = ["/data_images/Banana",
    #         "/data_images/Tomato"]
    #true_labels = ["banana", "tomato"]


    """
    data = extract_data(paths, true_labels, lim_data=200)
    im_shape = 50
    data.reshape(im_shape)            # making all data the same shape
    data.shuffle()
    #data.gray()

    #del data.data
    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.hot_vector,
                                                        train_size=0.8)

    X_train, X_test = X_train/255, X_test/255               # scaling image data
    """




    im_shape = 50
    #data_size = data.data[0].shape
    data_size = (im_shape, im_shape, 3)

    rec_field = 3
    filters = 20
    neuros_con = 50

    eta = 0.0001
    lmbd = 0.001
    epochs = 1
    batch_size = 10


    CNN = CNN_keras(input_shape=data_size,
                    receptive_field=rec_field,
                    n_filters = filters,
                    n_neurons_connected = neuros_con,
                    labels = true_labels,
                    eta = eta,
                    lmbd = lmbd)

    CNN.add_layer(show_model=False)


    """
    if im_shape is large (e.g. 200), max_data should be low (e.g. 500)
    if im_shape is low (e.g. 50), max_data can be large (e.g. 5000)
    """
    max_data = 5000
    lim_data = int(max_data/len(paths))
    #tot_data = 0
    lens = []
    for path in paths:
        #tot_data += len(os.listdir("./"+path))
        lens.append(len(os.listdir("./"+path)))
    min_len = np.min(lens)
    tot_data = min_len * len(paths)
    runs = int(tot_data / max_data)

    print("---------------------------------------------")
    print("Unique fruits:           ", len(paths))
    print("Total fruits:            ", tot_data)
    print("Image resolution:         %ix%i" % (im_shape, im_shape))
    print("Max data set to:         ", max_data)
    print("Total runs:              ", runs)
    print("Total fruit per run:     ", lim_data)
    print("---------------------------------------------")
    print()

    #runs = 1

    for i in range(runs):
        print("Run:   ", i+1)
        data = extract_data(paths, true_labels, lim_data=lim_data, from_data=i*lim_data)
        data.reshape(im_shape)            # making all data the same shape
        data.shuffle()
        #data.gray()

        #del data.data
        X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                            data.hot_vector,
                                                            train_size=0.8)

        X_train, X_test = X_train/255, X_test/255               # scaling image data



        CNN.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        scores = CNN.model.evaluate(X_test, y_test, verbose=1)

        data.delete_all_data()          # clear the memory
        print()






    # predicting on a image that have not been used in training or testing

    #test_im = np.array(Image.open("./test_images/test_apple.png"))
    test = extract_data(["/test_images"], ["tomato"])
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
