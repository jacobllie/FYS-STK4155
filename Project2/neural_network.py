import numpy as np
import matplotlib.pyplot as plt
import random as r

r.seed(100)
np.random.seed(100)

# If we want positive outputs, then we should use ReLU as activation function


class neural_network:

    def __init__(self, network, input=None):
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
        #self.activations[0] = np.array(input)
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

    def forward(self, input, act_func="sigmoid", act_func_out="ReLU"):
        """
        Activates the network by moving forward through the net.
        input: the inputs that the network will evaluate
        act_func: activation functions in the hidden layers
        act_func_out: activation function before the output layer
        """
        self.activations[0] = input
        for i in range(self.layers-1):
            self.activation_function(layer=i, act_func=act_func)
        #self.activation_function(layer=self.layers-2, act_func=act_func_out)



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



# testing with linear functions that are a*x + b + noise
# object is to make the NN estimate a and b
lines = 1000
data_points = 50
a =  np.array([r.randint(-5,5) for i in range(lines)])
b = np.array([r.randint(-2,2) for i in range(lines)])
x = np.linspace(0, 3, data_points)
all_lines = np.zeros((lines, data_points))

for i in range(lines):
    noise = 0.5*np.random.randn(data_points)
    all_lines[i,:] = a[i]*x + b[i] + noise

"""
# just plotting the individual data sets
plt.close()
plt.figure(figsize=(10,7))
for i in range(lines):
    plt.plot(x, all_lines[i], ".")
plt.grid()
plt.show()
"""

# setting up the neural network
#net = [len(all_lines.flatten()), 4, 2]           # define the size of the network
net = [2*data_points, 15, 2]           # define the size of the network
NN = neural_network(network=net)

# backtracking a certain number of times (equal the times we iplement test data
# on the neural network)
for i in range(lines-1):
    NN.forward(input=np.append(x, all_lines[i,:]))
    NN.back_propagation(target=[a[i], b[i]], eta=0.01)


# test data sent into the neural network
NN.forward(input=np.append(x, all_lines[-1,:]))
a_NN, b_NN = NN.activations[-1]


last_line = all_lines[-1,:]
plt.title("Trained the NN with %i data sets" % (lines-1))
plt.plot(x, last_line, ".", label="Test line w/ noise", color="red")
plt.plot(x, a[-1]*x+b[-1], label="Test line w/o noise (a=%i, b=%i)" % (a[-1], b[-1]), color="red")
plt.plot(x, a_NN*x+b_NN, label="NN line (a=%.3f, b=%.3f)" % (a_NN, b_NN), color="black")
plt.legend(); plt.grid(); plt.tight_layout()
plt.show()


#
