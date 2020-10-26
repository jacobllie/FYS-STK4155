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
        self.layers = len(network)
        self.activations = [[] for i in range(self.layers)]
        for i in range(self.layers):
            self.activations[i] = np.array([0 for j in range(network[i])])
        self.activations[0] = np.array(input)
        self.activations = np.array(self.activations)


        self.bias = [[] for i in range(self.layers-1)]
        self.weights = [[] for i in range(self.layers-1)]
        for i in range(self.layers-1):
            self.bias[i] = np.array([r.gauss(0,1) for j in range(network[i+1])])
            self.weights[i] = np.array([np.array([r.gauss(0,1) for k in range(network[i+1])]) for j in range(network[i])])

        self.weights = np.array(self.weights)
        self.bias = np.array(self.bias)

        #self.weights = np.array([np.random.randn(y, x) for x, y in zip(network[:-1], network[1:])])
        #print(self.weights)

    def forward(self, act_func="sigmoid", act_func_out="ReLU"):
        """
        Activates the network by moving forward through the net.
        act_func: activation functions in the hidden layers
        act_func_out: activation function before the output layer
        """
        for i in range(self.layers-2):
            self.activation_function(layer=i, act_func=act_func)
        self.activation_function(layer=self.layers-2, act_func=act_func_out)



    def activation_function(self, layer, act_func="sigmoid"):
        """
        layer: integer representing the respective layer in the network
        layer = 0 equals first hidden layer.
        act_func: the preferred activation function to use
        output: adjust the output values on the 'activations' corresponding the layer
        """

        z_node =  self.weights[layer].T @ self.activations[layer] + self.bias[layer]

        if act_func == "sigmoid":
            self.activations[layer+1] = self.sigmoid(z_node)

        elif act_func == "step_function" or act_func == "step function":
            self.activations[layer+1] = self.step_function(z_node)

        elif act_func == "tanh":
            self.activations[layer+1] = self.tanh(z_node)

        elif act_func == "ReLU":
            self.activations[layer+1] = self.ReLU(z_node)

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



    # following functions are part of the backtracking algorithm
    def da_dz(self, layer):
        return self.activations[layer] - self.activations[layer]**2

    # the derivative of the cost function with respect to weights at the output layer
    def dc_dw_L(self):
        return self.delta_L(target) * self.activations[-2] # -2 because we want the values from the last hidden layer

    # calculates the derivative of cost function with respect to the activations
    def dc_da_L(self, target):
        return self.activations[-1] - target

    # computing error in output layer
    def delta_L(self, target):
        return self.da_dz(self.layers-1) * self.dc_da_L(target)

    # computes error in the hidden layers
    def delta(self, layer, delta_front):
        deltas = np.zeros(self.network[layer])
        for i in range(len(deltas)):
            deltas[i] = deltas[i] + np.sum(delta_front * self.weights[layer][i] * self.activations[layer][i])
        return deltas



    # performing the back propagation algorithm to adjust bias and weights
    def back_propagation(self, target, eta=0.1):
        err = self.activations[1:]*0
        err[-1] = self.delta_L(target)
        for i in range(1, len(err)):
            lay = len(self.network) - i - 1
            err[-i-1] = self.delta(lay, err[-i])


        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                self.weights[i][j] = self.weights[i][j] - eta*err[i]*self.activations[i][j]
                #for k in range(self.weights[i].shape[1]):
                    #print(i, j, k, self.weights[i].shape)
                    #print(err[i], err[i].shape)
                    #print(self.activations[i], self.activations[i].shape)
                    #print()
                    #self.weights[i][j][k] = self.weights[i][j][k] - eta*err[i][k]*self.activations[i][j]

        self.bias = self.bias - eta*err






"""
inputs = [2.2, 3.1, -0.3]
targets = [5, 2]

net = [len(inputs), 4, 2]           # test of network size
NN = neural_network(net, inputs)
for i in range(len(net)-1):
    NN.activation_function(i)
NN.back_propagation(targets)targets
#print(NN.activations)
"""



# testing with linear funcitions that are to a*x + b + noise
# object is to make the algorithm estimate a and b
a = 0.5
b = 2
x = np.linspace(0,3,2)
lines = 10
params = np.ones((2,lines))
params[0,:], params[1,:] = a, b
params = params + 0.1*np.random.randn(2,lines)


# just plotting the individual lines
plt.figure(figsize=(10,7))
for i in range(len(params[0])):
    plt.plot(x, params[0][i]*x+params[1][i])

# setting up the neural network
net = [len(params.flatten()), 5, 2]           # define the size of the network
NN = neural_network(network=net, input=params.flatten())


# runs the neural network once
NN.forward(act_func_out="sigmoid")
a_NN, b_NN = NN.activations[-1]
plt.plot(x, a_NN*x + b_NN, "--", color="red",
        label="Before BP (a=%.3f, b=%.3f)" % (a_NN, b_NN))


# backtracking the number of times we have data (lines)
for back_tracks in range(lines):
    A = params[0, back_tracks]
    B = params[1, back_tracks]
    NN.back_propagation([A,B], eta=0.1)#/(1+back_tracks))
    NN.forward(act_func_out="sigmoid")
    #a_NN, b_NN = NN.activations[-1]
    #plt.plot(x, a_NN*x + b_NN, label="After %i BT's" % (back_tracks+1))

a_NN, b_NN = NN.activations[-1]
plt.plot(x, a_NN*x+b_NN, "--", color="black",
        label="After %i BP's (a=%.3f, b=%.3f)" % (back_tracks+1, a_NN, b_NN))
plt.grid(); plt.legend()
#print(test_target[-1]-test_target[0], (a_NN*x+b_NN)[-1]-(a_NN*x+b_NN)[0])
plt.tight_layout();plt.show()





#
