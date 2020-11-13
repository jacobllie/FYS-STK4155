import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import datasets
import sys
from sklearn.metrics import confusion_matrix
import time

from activation_function import sigmoid, softmax, identity, tanh, relu
from functions import FrankeFunction
from data_prep import data_prep
from NN import DenseLayer, NN
from cost_functions import accuracy, MSE, CE

#plt.rcParams.update({'font.size': 14})

# download MNIST dataset
data = datasets.load_digits()
labels = data.target.reshape(-1,1)
N = labels.shape[0]
inputs = data.images.reshape(N,-1)
features = inputs.shape[1]

ind = np.arange(0,N)
random.shuffle(ind)

X = inputs[ind]
Y = labels[ind]

X_test = X[:int(0.3*N)]
X_train = X[int(0.3*N):]
Y_test = Y[:int(0.3*N)]
Y_train = Y[int(0.3*N):]

for i in range(10):
    print("# of %d: %d" %(i,np.sum(Y_train == i)))

one_hot = np.zeros((Y_train.shape[0], 10))
for i in range(Y_train.shape[0]):
    one_hot[i,Y_train[i]] = 1

mse = MSE()
ce = CE()
m = X_train.shape[0]
batch = np.arange(0,m)

def epoch(eta=0.04, penalty=0.4, epochs=200, mini_batch_size = 100, t0=5, t1=50,
        create_conf=False):
    layer1 = DenseLayer(features, 100, sigmoid())
    #layer2 = DenseLayer(100, 50, sigmoid())
    #layer3 = DenseLayer(100, 50, sigmoid())
    layer4 = DenseLayer(100, 10, softmax())

    layers = [layer1, layer4]
    network = NN(layers)
    cost_array = np.zeros((epochs,2))
    def learning_schedule(t):
        return 0.04#t0/(t+t1)
    for i in range(epochs):
        random.shuffle(batch)
        X_train_shuffle = X_train[batch]
        one_hot_shuffle = one_hot[batch]
        Y_train_shuffle = Y_train[batch]
        #eta = learning_schedule(i)
        network.SGD(ce, 100, X_train_shuffle, one_hot_shuffle, eta, penalty)
        Y_pred = np.argmax(network.feedforward(X_test), axis=1)
        Y_pred_train = np.argmax(network.feedforward(X_train_shuffle), axis=1)
        cost_array[i,0] = accuracy()(Y_test.ravel(), Y_pred)
        cost_array[i,1] = accuracy()(Y_train_shuffle.ravel(), Y_pred_train)
    print("accuracy on train data = %.3f" %cost_array[-1, 1])
    print("accuracy on test data = %.3f" %cost_array[-1, 0])
    if create_conf == True:
        #creating confusion matrix
        numbers = np.arange(0,10)
        conf_matrix = confusion_matrix(Y_pred,Y_test,normalize="true")
        heatmap = sb.heatmap(conf_matrix,cmap="viridis",
                              xticklabels=["%d" %i for i in numbers],
                              yticklabels=["%d" %i for i in numbers],
                              cbar_kws={'label': 'Accuracy'},
                              fmt = ".2",
                              edgecolor="none",
                              annot = True)
        heatmap.set_xlabel("pred")
        heatmap.set_ylabel("true")

        heatmap.set_title(r"FFNN prediction accuracy with $\lambda$ = {:.1e} $\eta$ = {:.1e}"\
            .format(penalty, eta))
        fig = heatmap.get_figure()
        fig.savefig("../figures/MNIST_confusion_net.pdf",bbox_inches='tight',
                                  pad_inches=0.1,
                                  dpi = 1200)
        plt.show()
    return cost_array[-1]

create_heatmap = input("Analyse penalty parameter and learning rate [Y/n]: ")
if create_heatmap == "Y" or create_heatmap == "y":
    create_heatmap = True
elif create_heatmap == "N" or create_heatmap == "n":
    create_heatmap = False
else:
    print("Please input Y or n!")
    sys.exit()

if create_heatmap == True:
    #Make an accuracy map for different etas and lambdas
    penalties = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    accuracy_map = np.zeros((len(penalties), len(etas), 2))

    start_time = time.time()
    for i, penalty in enumerate(penalties):
        for k, eta in enumerate(etas):
            print("-----------------------------")
            accuracy_map[i, k] = epoch(eta, penalty)
    print("--- %s seconds ---" % (time.time() - start_time))

    heatmap = sb.heatmap(accuracy_map[:,:,0],cmap="viridis",
                                  xticklabels=etas,
                                  yticklabels=penalties,
                                  cbar_kws={'label': 'Accuracy'},
                                  fmt = ".3",
                                  annot = True)
    plt.yticks(rotation=0)
    heatmap.set_xlabel(r"$\eta$")
    heatmap.set_ylabel(r"$\lambda$")
    heatmap.invert_yaxis()
    heatmap.set_title("Accuracy on MNIST with FFNN")
    fig = heatmap.get_figure()
    plt.show()
    fig.savefig("../figures/MNIST_heatmap_CE.pdf", bbox_inches='tight',
                                                pad_inches=0.1)

epoch(eta=0.1, penalty=1, epochs=200, create_conf=True)
