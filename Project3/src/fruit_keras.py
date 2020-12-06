import os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from data_adjustment import extract_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cost_functions import accuracy

def NN_keras(input, hidden_layers, output, act_func, eta, penalty=0):
    """
    Function for creating a neural network with Keras
    ------------
    input: features of data, integer
    hidden_layers: list with neurons
    output: features of output, integer
    eta: learning rate
    penalty: penalty/regularization parameter
    """

    model = Sequential()
    model.add(Dense(input, activation=act_func, kernel_regularizer=l2(penalty)))
    for neurons in layers:
        model.add(Dense(neurons, activation=act_func,
                  kernel_regularizer=l2(penalty)))

    model.add(Dense(output, activation="softmax"))

    sgd = SGD(lr=eta)
    model.compile(loss="categorical_crossentropy", optimizer=sgd)
    return model


if __name__ == '__main__':

    paths = ["../images/Apple/Total Number of Apples",
             "../images/Banana"]
             #"../images/Kiwi/Total Number of Kiwi fruit",
             #"../images/Mango",
             #"../images/Orange",
             #"../images/Pear",
             #"../images/Tomato"]
    true_labels = ["apple", "banana"]#, "kiwi", "mango",
                   #"orange", "pear", "tomato"]

    #paths = ["/data_images/Banana",
    #         "/data_images/Tomato"]
    #true_labels = ["banana", "tomato"]

    num_fruits = len(true_labels)

    im_shape = 50
    #data_size = data.data[0].shape
    data_size = (im_shape, im_shape, 3)

    rec_field = 3
    filters = 20
    neuros_con = 50

    eta = 0.0005
    lmbd = 0.001
    epochs = 2
    batch_size = 5

    """
    if im_shape is large (e.g. 200), max_data should be low (e.g. 500)
    if im_shape is low (e.g. 50), max_data can be large (e.g. 5000)
    """
    max_data = 500
    lim_data = int(max_data/len(paths))
    #tot_data = 0
    lens = []
    for path in paths:
        #tot_data += len(os.listdir("./"+path))
        lens.append(len(os.listdir("./"+path)))
    min_len = np.min(lens)
    print(min_len)
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

    layers = [3000, 1000, 200, 10]

    NN = NN_keras(input = im_shape*im_shape, hidden_layers = layers,
                  output=num_fruits, act_func = "relu", eta = eta, penalty = lmbd)

    acc_score = accuracy()

    for i in range(runs):
        print("Run:   ", i+1)
        data = extract_data(paths, true_labels, lim_data=lim_data,
                            from_data=i*lim_data)
        data.reshape(im_shape)            # making all data the same shape
        data.shuffle()
        data.gray()
        data.flatten()

        #del data.data
        X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                            data.hot_vector,
                                                            train_size=0.8)

        X_train, X_test = X_train/255, X_test/255           # scaling image data

        NN.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
               verbose=1)
        prediction = np.argmax(NN.predict_step(X_test), axis=1)
        score = acc_score(np.argmax(y_test, axis=1), prediction)
        data.delete_all_data()          # clear the memory
        print("Accuracy = {:.4}".format(score))

    # predicting on a image that have not been used in training or testing

    test = extract_data(["../images/test_images"], ["banana"])
    test.reshape(im_shape)         # making all data the same shape
    test.shuffle()
    test.gray()
    test.flatten()
    pred = np.argmax(NN.predict_step(test.data))
    print(NN.predict_step(test.data))
    plt.figure(figsize=[12,6])
    plt.subplot(121)
    plt.tight_layout()
    plt.imshow(test.data.reshape((50,50)), cmap="gray")
    plt.title("CNN predicts: %s" %(true_labels[pred]))
    plt.subplot(122)
    plt.imshow(test.real_data[-1,...], cmap="gray")
    plt.title("Real data %s" %(test.labels[0]))
    plt.tight_layout()
    plt.savefig("Keras_NN.pdf")
    plt.show()
