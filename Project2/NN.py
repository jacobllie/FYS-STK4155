from numpy import random
from activation_function import sigmoid, identity
import numpy as np
from numpy import random

random.seed(100)

class dense_layer:
    def __init__(self, inputs, outputs, act_func):
        self.weights = 1e-3*random.randn(inputs, outputs)
        self.b = random.randn(1, outputs)
        self.act_func = act_func
        #self.a, self.da, self.z = None

    def __call__(self, X):
        #calculate activation
        self.z = X@self.weights + self.b
        self.a = self.act_func(self.z)
        #calculate derivative
        self.da = self.act_func.deriv(self.z)
        return self.a

class NN:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, a):
        #feed forward
        for layer in self.layers:
            a = layer(a)
        return a

    def backprop(self, cost, x, y, eta):
        L = self.layers
        a = L[-1].a
        delta_l = (L[-1].weights@((a - y)*L[-1].act_func.deriv(L[-1].z)).T).T
        for i in reversed(range(1, len(L)-1)):
            delta_l = delta_l*L[i].act_func.deriv(L[i].z)
            L[i].weights = L[i].weights - (eta*delta_l.T@L[i-1].a).T
            L[i].b = L[i].b - eta*delta_l
            delta_l = delta_l@L[i].weights.T
        #First hidden layer
        delta_l = delta_l*L[0].act_func.deriv(L[0].z)
        L[0].weights = L[0].weights - (eta*delta_l.T@x).T
        L[0].b = L[0].b - eta*delta_l
        delta_l = delta_l@L[0].weights.T

class MSE:
    def __call__(self, y, a):
        return 1/len(y)*np.sum(y - a)**2

    def deriv(self, y, a):
        return 2/len(y)*(a - y)

#define cost function

x = np.linspace(0,1,100).reshape(-1,1) + 0.01*random.randn(100, 1)
y = x**2
layer1 = dense_layer(1, 50, sigmoid())
layer2 = dense_layer(50, 20, sigmoid())
layer3 = dense_layer(20, 1, identity())

layers = [layer1, layer2, layer3]
network = NN(layers)
network(x)
mse = MSE()
print(mse(y, layer3.a))
for i in range(1000):
    network.backprop(MSE, x, y, 1)
    network(x)
print(mse(y, layer3.a))
