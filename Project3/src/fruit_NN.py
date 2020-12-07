import os
import pickle

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from Network import DenseLayer, NN
from data_adjustment import extract_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cost_functions import accuracy, CE
from activation_function import relu, softmax
from sklearn.metrics import confusion_matrix
from tqdm import trange

if __name__ == '__main__':

    color_input = input("Color or Black-white [C/Bw]: ")
    im_shape = int(input("Size of N images (N x N pixels): "))

    if color_input == "C" or color_input == "c":
        color_scale = False
    elif color_input == "Bw" or color_input == "bw":
        color_scale = True
    else:
        raise ValueError("Invalid input argument")

    paths = ["../images/Apple/",
             "../images/Banana",
             "../images/Kiwi",
             "../images/Mango",
             "../images/Orange",
             "../images/Pear",
             "../images/Tomatoes"]

    true_labels = ["apple", "banana", "kiwi", "mango", "orange", "pear",
                   "tomato"]

    num_fruits = len(true_labels)

    data_size = (im_shape, im_shape, 3)

    eta = 0.0001
    lmbd = 0.001
    epochs = 5
    batch_size = 5

    """
    if im_shape is large (e.g. 200), max_data should be low (e.g. 500)
    if im_shape is low (e.g. 50), max_data can be large (e.g. 5000)
    """
    max_data = 500
    lim_data = int(max_data/len(paths))
    print(lim_data)
    lens = []
    for path in paths:
        lens.append(len(os.listdir("./"+path)))
    min_len = np.min(lens)

    tot_data = min_len * len(paths)
    runs = int(tot_data / max_data) - 1

    print("---------------------------------------------")
    print("Unique fruits:           ", len(paths))
    print("Total fruits:            ", tot_data)
    print("Image resolution:         %ix%i" % (im_shape, im_shape))
    print("Max data set to:         ", max_data)
    print("Total runs:              ", runs)
    print("Total fruit per run:     ", lim_data)
    print("---------------------------------------------")

    ce = CE()
    acc_score = accuracy()
    scaler = StandardScaler()

    if color_scale:
        layer1 = DenseLayer(im_shape*im_shape, 3000, relu(), Glorot=True)
    else:
        layer1 = DenseLayer(im_shape*im_shape*3, 3000, relu(), Glorot=True)
    layer2 = DenseLayer(3000, 1000, relu(), Glorot=True)
    layer3 = DenseLayer(1000, 200, relu(), Glorot=True)
    layer4 = DenseLayer(200, 10, relu(), Glorot=True)
    layer5 = DenseLayer(10, num_fruits, softmax())

    layers = [layer1, layer2, layer3, layer4, layer5]
    network = NN(layers, ce)

    for i in trange(20):
        print("Run:   ", i+1)
        data = extract_data(paths, true_labels, lim_data=lim_data,
                            from_data=i*lim_data)
        data.reshape(im_shape)            # making all data the same shape
        if color_scale:
            data.gray()
        data.flatten()

        #del data.data
        X_train, X_valid, y_train, y_valid = train_test_split(data.data,
                                                            data.hot_vector,
                                                            train_size=0.8)
        if i==0:
            scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)

        for i in range(epochs):
            network.SGD(batch_size, X_train, y_train, eta, lmbd)

        prediction = np.argmax(network.feedforward(X_valid), axis=1)
        true_label = np.argmax(y_valid, axis=1)

        score = acc_score(true_label, prediction)
        data.delete_all_data()          # clear the memory
        print("Accuracy = {:.4}".format(score))

    print("-------------------------")
    print("FINISHED TRAINING NETWORK")
    print("-------------------------")

    with open('network_NN_'+color_input+'.pkl', 'wb') as output:
        pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)

    # predicting on a image that have not been used in training or testing

    test_network = extract_data(paths, true_labels, lim_data=lim_data,
                        from_data=runs*lim_data)
    test_network.reshape(im_shape)            # making all data the same shape
    if color_scale:
        test_network.gray()
    test_network.flatten()

    test_images = test_network.data
    test_labels = test_network.labels
    one_hot = test_network.hot_vector
    test_images = scaler.transform(test_images)

    test_prediction = np.argmax(network.feedforward(test_images), axis=1)
    test_num_label = np.argmax(one_hot, axis=1)

    print("Accuracy on test data: %.2f" %acc_score(test_prediction,
                                                   test_num_label))

    indices = np.random.randint(len(test_prediction), size=5)
    for ind in indices:
        plt.figure(figsize=[12,6])
        plt.subplot(121)
        plt.tight_layout()
        if color_scale:
            plt.imshow(test_images[ind].reshape((im_shape, im_shape)),
                       cmap="gray")
        else:
            plt.imshow(test_images[ind].reshape((im_shape, im_shape,3)),
                       cmap="gray")
        plt.title("NN predicts: %s" %(true_labels[test_prediction[ind]]))
        plt.subplot(122)
        plt.imshow(test_network.real_data[ind], cmap="gray")
        plt.title("Real data %s" %(test_labels[ind]))
        plt.tight_layout()
        plt.savefig("../results/NN/NN_"+color_input+str(ind)+".pdf",
                    bbox_inches='tight',
                    pad_inches=0.1)
        plt.show()

    conf_matrix = confusion_matrix(test_prediction, test_num_label,
                                   normalize="true")

    heatmap = sb.heatmap(conf_matrix, cmap="viridis",
                         xticklabels=[label for label in true_labels],
                         yticklabels=[label for label in true_labels],
                         cbar_kws={'label': 'Accuracy'},
                         fmt = ".2",
                         edgecolor="none",
                         annot = True)

    heatmap.set_xlabel("pred")
    heatmap.set_ylabel("true")

    heatmap.set_title(r"Neural Network prediction of fruit")
    fig = heatmap.get_figure()
    plt.yticks(rotation=0)
    fig.savefig("../results/NN/NN_"+color_input+".pdf",
                bbox_inches='tight',
                pad_inches=0.1)
    plt.show()
