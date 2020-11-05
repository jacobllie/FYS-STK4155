import numpy as np
import matplotlib.pyplot as plt

from numpy import random
from sklearn import datasets

from activation_function import sigmoid, softmax, identity
from functions import FrankeFunction
from data_prep import data_prep
from NN import dense_layer, NN
from cost_functions import accuracy, CE, MSE
from sklearn.preprocessing import StandardScaler

# download MNIST dataset
data = datasets.load_digits()
labels = data.target.reshape(-1,1)
N = labels.shape[0]
inputs = data.images.reshape(N,-1)
features = inputs.shape[1]

#prep data
ind = np.arange(0,N)
random.shuffle(ind)
X = inputs[ind]         #input
Y = labels[ind]         #output

"""
data = data_prep()
data.MNIST(data,inputs,labels)
X_train,X_test,Y_train,Y_test = data.train_test_split_scale()

ind = np.arange(0,len(X_train))
random.shuffle(ind)
X_train = inputs[ind]         #input
Y_train = labels[ind]
"""



#split in train and test
X_test = X[:int(0.3*N)]
X_train = X[int(0.3*N):]
Y_test = Y[:int(0.3*N)]
Y_train = Y[int(0.3*N):]


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#set up one-hot vector
one_hot = np.zeros((Y_train.shape[0], 10))
for i in range(Y_train.shape[0]):
    one_hot[i,Y_train[i]] = 1

#Make neural network with zero hidden layer
output_layer = dense_layer(features, 10, softmax())

layers = [output_layer]
log_net = NN(layers)
cost_func = CE()    #using cross-entropy as cost function

epochs = 500
mini_batch_size = N//200
eta = 0.001
m = X_train.shape[0]

ind = np.arange(0, X_train.shape[0])
cost_array = np.zeros(epochs)
for i in range(epochs):     #looping epochs
    random.shuffle(ind)
    X_train = X_train[ind]
    one_hot = one_hot[ind]
    for j in range(0,m,mini_batch_size):
        log_net.logreg_backprop(cost_func, X_train[j:j+mini_batch_size],
            one_hot[j:j+mini_batch_size], eta)
    Y_pred = np.argmax(log_net.feed_forward(X_test), axis=1)
    #print(Y_pred)
    #print(Y_test)
    cost_array[i] = accuracy()(Y_test, Y_pred)*100
plt.plot(cost_array)
plt.show()

"""
for i in range(len(Y_test)):
    print(Y_test[i],log_net.logreg_backprop(cost_func,X_train[i],one_hot[i],eta))
"""
