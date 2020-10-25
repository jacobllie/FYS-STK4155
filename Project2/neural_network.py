import numpy as np
import matplotlib.pyplot as plt
import random as r

r.seed(100)
np.random.seed(100)

# If we want positive outputs, then we should use ReLU as activation function


class neural_network:

    def __init__(self, network, input):
        """
        'network' is a list where its length defines the number of layers
        and the values of the elements represents the number of neurons in
        each layer. For example network = [3,4,2] has 3 layers where the first
        layer (input layer) contain 3 neurons, second layer (hidden layer)
        contain 4 neurons and the last layer (output layer) contain 2 neurons.
        input are the values inserted to the neural network.
        """

        self.network = network
        self.grid = [[] for i in range(len(network))]
        for i in range(len(network)):
            self.grid[i] = np.array([0 for j in range(network[i])])
        self.grid[0] = np.array(input)
        self.grid = np.array(self.grid)


        self.layers = len(network)
        self.bias = [[] for i in range(self.layers-1)]
        self.weights = [[] for i in range(self.layers-1)]
        for i in range(self.layers-1):
            self.bias[i] = np.array([r.gauss(0,1) for j in range(network[i+1])])
            self.weights[i] = np.array([np.array([r.gauss(0,1) for k in range(network[i+1])]) for j in range(network[i])])

        self.weights = np.array(self.weights)
        self.bias = np.array(self.bias)

        #self.weights = np.array([np.random.randn(y, x) for x, y in zip(network[:-1], network[1:])])
        #print(self.weights)


    def activation_function(self, layer, act_func="sigmoid"):
        """
        layer: integer representing the respective layer in the network
        layer = 0 equals first hidden layer.
        act_func: the preferred activation function to use
        output: adjust the output values on the grid corresponding the layer
        """

        z_node =  self.weights[layer].T @ self.grid[layer] + self.bias[layer]

        if act_func == "sigmoid":
            self.grid[layer+1] = self.sigmoid(z_node)

        elif act_func == "step_function" or act_func == "step function":
            self.grid[layer+1] = self.step_function(z_node)

        elif act_func == "tanh":
            self.grid[layer+1] = self.tanh(z_node)

        elif act_func == "ReLU":
            self.grid[layer+1] = self.ReLU(z_node)

        else:
            msg = "Need to define activation function! (%s) is illegal. Try sigmoid, step function, tanh, ReLU..." % act_func
            assert False, msg


    # activation functions (sigmoid, step function, tanh, ReLU)
    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def step_function(self, t):
        # all positive/negative values are set equal to 1/0
        if len(t) == 1:
            if t > 0:
                return 1
            else:
                return 0

        else:
            pos = np.where(t > 0)
            t[:] = 0
            t[pos] = 1
            return t

    def tanh(self, t):
        return np.tanh(t)

    def ReLU(self, t):
        # all negative values are set equal to zero
        if len(t) == 1:
            return max(0, t)
        else:
            t[np.where(t < 0)] = 0
            return t


    def cost_func(self, target, outputs, method="MSE"):
        if method == "MSE":
            test = 0


    def da_dz(self, node, step_func="sigmoid"):
        if step_func == "sigmoid":
            act = sigmoid(node)
        if step_func == "step_function" or step_func == "step function":
            act = step_function(node)
        if step_func == "tanh":
            act = tanh(node)
        if step_func == "ReLU":
            act = ReLU(node)

        return act - act**2


    def back_propagation(self):
        test = 0






inputs = [2.2, 3.1, -0.3]
net = [len(inputs), 4, 2]           # test of network size
NN = neural_network(net, inputs)
#NN.activation_function(0, "sigmoid")
#NN.activation_function(1, "sigmoid")
#print(NN.grid)
