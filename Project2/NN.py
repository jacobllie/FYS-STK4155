from numpy import random
from activation_function import sigmoid, identity
import numpy as np
from numpy import random

random.seed(100)

class dense_layer:
    def __init__(self, inputs, outputs, act_func):
        self.weights = random.randn(inputs, outputs)
        self.b = random.randn(1, outputs)
        self.act_func = act_func

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

    def feed_forward(self, a):
        for layer in self.layers:
            a = layer(a)
        return a

    def backprop(self, cost, x, y, eta):
        L = self.layers
        a = L[-1].a
        delta_l = cost.deriv(y, a)*L[-1].da
        for i in reversed(range(1, len(L)-1)):
            delta_l = (delta_l@L[i+1].weights.T)*L[i].da
            L[i].weights = L[i].weights - eta*(L[i-1].a.T@delta_l)
            L[i].b = L[i].b - eta*delta_l
        #First hidden layer
        delta_l = (delta_l@L[1].weights.T)*L[0].da
        L[0].weights = L[0].weights - eta*(x.T@delta_l)
        L[0].b = L[0].b - eta*delta_l
        self.feed_forward(x)

class MSE:
    def __call__(self, y, a):
        return 1/len(y)*np.sum(a - y)**2

    def deriv(self, y, a):
        return 2/len(y)*(a - y)

x = np.linspace(0,1,100).reshape(-1,1) + 0.1*random.randn(100,1)
y = x**2
layer1 = dense_layer(1, 10, sigmoid())
layer2 = dense_layer(10, 10, sigmoid())
layer3 = dense_layer(10, 20, sigmoid())
layer4 = dense_layer(20, 5, sigmoid())
layer5 = dense_layer(5, 1, identity())

layers = [layer1, layer2, layer3, layer4, layer5]
network = NN(layers)
network.feed_forward(x)
mse = MSE()
print(mse(y, layers[-1].a))
for i in range(1000):
    network.backprop(MSE(), x, y, 0.5)
print(mse(y, layers[-1].a))
