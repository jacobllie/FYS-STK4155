from numpy import random
from activation_function import sigmoid

class dense_layer:
    def __init__(self, inputs, outputs, act_func):
        self.weights = 1e-3*random.randn(inputs, outputs)
        self.b = random.randn(1, outputs)
        self.act_func = act_func

    def __call__(self, X):
        #feed forward
        self.a = self.act_func(X@self.weights + self.b)
        #calculate derivative
        self.da = self.act_func.deriv(X@self.weights + self.b)
        return self.a

class NN:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backprop():
        pass

#define cost function

x = random.randn(100, 15)
layer1 = dense_layer(15, 50, sigmoid())
layer2 = dense_layer(50, 20, sigmoid())
layer3 = dense_layer(20, 1, sigmoid())

layers = [layer1, layer2, layer3]
#print(layer1(x).shape)
network = NN(layers)
print(network(x))
