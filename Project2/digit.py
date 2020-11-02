import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn import datasets
from activation_function import sigmoid,softmax,identity
from functions import FrankeFunction
from data_prep import data_prep
from NN import dense_layer,NN,MSE
from activation_function import sigmoid,softmax,identity
from cost_functions import accuracy

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
#print(mse(Y_test, network.feed_forward(X_test)))
network.feed_forward(X_train)  #sending in x and y from design matrix.
for i in range(500):
    network.backprop(mse, X_train, one_hot, 0.1)

#Test the network on the test data
Y_tilde = network.feed_forward(X_test)
#print(mse(Y_test, Y_tilde))

for i in range(50):
    print("|    %d   |   %d  |" %(Y_test[i], np.argmax(Y_tilde[i])))

# choose some random images to display
indices = np.arange(len(inputs))
random_indices = np.random.choice(indices, size=5)

for i, image in enumerate(data.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % data.target[random_indices[i]])
plt.show()
