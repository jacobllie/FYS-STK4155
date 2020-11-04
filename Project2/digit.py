import numpy as np
from numpy import random
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

one_hot = np.zeros((Y_train.shape[0], 10))
for i in range(Y_train.shape[0]):
    one_hot[i,Y_train[i]] = 1

layer1 = dense_layer(features, 100, sigmoid())
layer2 = dense_layer(100, 80, sigmoid())
layer3 = dense_layer(80, 50, sigmoid())
layer4 = dense_layer(50, 20, sigmoid())
layer5 = dense_layer(20, 10, softmax())

layers = [layer1, layer2, layer3, layer4, layer5]
network = NN(layers)
mse = MSE()

cost_array = np.zeros((1000,2))
for i in range(1000):
    network.backprop(mse, X_train, one_hot, 0.5, 5)
    Y_pred = np.argmax(network.feed_forward(X_test), axis=1)
    Y_pred_train = np.argmax(network.feed_forward(X_train), axis=1)
    cost_array[i,0] = accuracy()(Y_test.ravel(), Y_pred)*100
    cost_array[i,1] = accuracy()(Y_train.ravel(), Y_pred_train)*100
plt.plot(cost_array)
plt.show()

#Test the network on the test data
Y_tilde = network.feed_forward(X_test)
pred = np.argmax(Y_tilde, axis=1)
#print(mse(Y_test, Y_tilde))

print("accuracy = %.3f" %(accuracy()(Y_test.ravel(), pred)*100))

# choose some random images to display

for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(data.images[ind][i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" %data.target[ind][i])
    plt.text(1, -4, "Pred: %d" %pred[i], fontsize=16)

plt.show()
