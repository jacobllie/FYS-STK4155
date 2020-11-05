import numpy as np
from numpy import random
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import datasets
from activation_function import sigmoid,softmax,identity
from functions import FrankeFunction
from data_prep import data_prep
from NN import dense_layer, NN
from cost_functions import accuracy, MSE

plt.rcParams.update({'font.size': 14})

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

layer1 = dense_layer(features, 100, sigmoid())
layer2 = dense_layer(100, 20, sigmoid())
layer3 = dense_layer(80, 50, sigmoid())
layer4 = dense_layer(50, 20, sigmoid())
layer5 = dense_layer(20, 10, softmax())


layers = [layer1, layer2, layer5]
mse = MSE()
m = X_train.shape[0]
batch = np.arange(0,m)

def epoch(eta, penalty, epochs=200, mini_batch_size = 100):
    network = NN(layers)
    cost_array = np.zeros((epochs,2))
    for i in range(epochs):
        random.shuffle(batch)
        X_train_shuffle = X_train[batch]
        one_hot_shuffle = one_hot[batch]
        Y_train_shuffle = Y_train[batch]
        for j in range(0, m, mini_batch_size):
            network.backprop(mse, X_train_shuffle[j:j+mini_batch_size],
                one_hot_shuffle[j:j+mini_batch_size], eta, penalty)
        Y_pred = np.argmax(network.feed_forward(X_test), axis=1)
        Y_pred_train = np.argmax(network.feed_forward(X_train_shuffle), axis=1)
        cost_array[i,0] = accuracy()(Y_test.ravel(), Y_pred)
        cost_array[i,1] = accuracy()(Y_train_shuffle.ravel(), Y_pred_train)
    """plt.plot(cost_array[:,1], label="Train")
    plt.plot(cost_array[:,0], label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()"""
    print("accuracy on train data = %.3f" %cost_array[-1, 1])
    print("accuracy on test data = %.3f" %cost_array[-1, 0])
    return cost_array[-1]

penalties = np.logspace(-2,0,11)
etas = np.logspace(-3,-1,11)
accuracy_map = np.zeros((len(penalties), len(etas), 2))

for i, penalty in enumerate(penalties):
    for k, eta in enumerate(etas):
        accuracy_map[k, i] = epoch(eta, penalty)

np.save("accuracy_map.npy", accuracy_map)

def plot_pictures(k):
    #k determines which pictures to plot
    # choose some random images to display
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(data.images[ind][i+k], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %d" %data.target[ind][i+k])
        plt.text(1, -4, "Pred: %d" %Y_pred[i+k], fontsize=16)
    plt.show()
