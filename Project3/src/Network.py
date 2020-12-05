import numpy as np
from numpy import random

class DenseLayer:
    def __init__(self, inputs, outputs, act_func, Glorot=False):
        np.random.seed(200)
        stddev=1
        if Glorot:
            #try with Glorot inializations of weights
            variance = 2.0/(inputs + outputs)
            stddev = np.sqrt(variance)
        self.weights = stddev*random.randn(inputs, outputs)
        self.b = random.randn(1, outputs)
        self.act_func = act_func

    def __call__(self, X):
        self.z = X @ self.weights + self.b
        #print(self.z)
        self.a = self.act_func(self.z)
        self.da = self.act_func.deriv(self.z)
        return self.a

class NN:
    def __init__(self, layers, cost):
        self.cost = cost
        self.layers = layers

    def feedforward(self, a):
        for layer in self.layers:
            a = layer(a)
        return a

    def backprop(self, x, y, eta, penalty=0):
        self.feedforward(x)  #using the updated weights and biases to get new output layer
        #Starting with output layer
        L = self.layers
        a = L[-1].a
        delta_l = self.cost.deriv(y, a)*L[-1].da
        #Looping over layer from output to 1st hidden
        for i in reversed(range(1, len(L)-1)):      #we only backprop until L - 2 layer
            delta_l = (delta_l @ L[i+1].weights.T) * L[i].da
            #Updating weights
            L[i].weights = L[i].weights - eta*(L[i-1].a.T @ delta_l) \
                - eta*penalty*L[i].weights/len(y)
            #Updating biases
            L[i].b = L[i].b - eta * delta_l[0, :]
        #Updating the first hidden layer with the new weights and biases.
        delta_l = (delta_l @ L[1].weights.T) * L[0].da
        L[0].weights = L[0].weights - eta*(x.T @ delta_l) \
            - eta*penalty*L[0].weights/len(y)
        L[0].b = L[0].b - eta*delta_l[0,:]


    def SGD(self, mini_batch_size, X_train_shuffle, z_train_shuffle, eta, penalty=0):
        for j in range(0,X_train_shuffle.shape[0],mini_batch_size):
            self.backprop(X_train_shuffle[j:j+mini_batch_size],
                z_train_shuffle[j:j+mini_batch_size],eta,penalty)
        return X_train_shuffle, z_train_shuffle
