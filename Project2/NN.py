from numpy import random
from activation_function import sigmoid, identity, relu, leaky_relu,softmax
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from data_prep import data_prep
from functions import FrankeFunction
from cost_functions import MSE

random.seed(100)

class dense_layer:
    def __init__(self, inputs, outputs, act_func):
        self.weights = random.randn(inputs, outputs)
        self.b = random.randn(1, outputs)
        self.act_func = act_func

    def __call__(self, X):
        self.z = X @ self.weights + self.b
        self.a = self.act_func(self.z)
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
        self.feed_forward(x)  #using the updated weights and biases to get new output layer
        #Starting with output layer
        L = self.layers
        a = L[-1].a
        delta_l = cost.deriv(y, a)*L[-1].da
        #Looping over layer from output to 1st hidden
        for i in reversed(range(1, len(L)-1)):      #we only backprop until L - 2 layer
            delta_l = (delta_l @ L[i+1].weights.T) * L[i].da
            #Updating weights
            L[i].weights = L[i].weights - eta*(L[i-1].a.T @ delta_l)
            #Updating biases
            L[i].b = L[i].b - eta * delta_l[0,:]
        #Updating the first hidden layer with the new weights and biases.
        delta_l = (delta_l @ L[1].weights.T) * L[0].da
        L[0].weights = L[0].weights - eta*(x.T @ delta_l)
        L[0].b = L[0].b - eta*delta_l[0,:]

    def logreg_backprop(self, cost, x, y, eta):
        self.feed_forward(x)  #using the updated weights and biases to get new output layer
        #Starting with output layer
        L = self.layers
        a = L[0].a
        delta_l = cost.deriv(y, a)*L[0].da
        #Updating output layer with the new weights and biases.
        L[0].weights = L[0].weights - eta*(x.T @ delta_l)
        L[0].b = L[0].b - eta*delta_l[0,:]


#Test case
if __name__ == "__main__":

    epochs = 500
    n = 100
    noise = 0.1
    eta = 0.05
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    x,y = np.meshgrid(x,y)
    z = np.ravel(FrankeFunction(x,y) + noise*np.random.randn(n,n))

    data = data_prep()
    X = data.X_D(x,y,z,1)    #X_train,X_test,z_train,z_test
    train_input,test_input,train_output,test_output = data.train_test_split_scale()

    train_output = np.reshape(train_output,(-1,1))               #the shape was (x,) we needed (x,1) for obvious reasons
    test_output = np.reshape(test_output,(-1,1))
    x_train = train_input[:, [1,2]]
    x_test = test_input[:, [1,2]]

    #Setting up network
    layer1 = dense_layer(2, 10, sigmoid())
    layer2 = dense_layer(10, 6, sigmoid())
    layer3 = dense_layer(6, 1, identity())
    layers = [layer1, layer2, layer3]
    network = NN(layers)
    #Finding MSE on untrained network
    mse = MSE()
    print("Test MSE before training network: %.4f" %mse(test_output, network.feed_forward(x_test)))
    #Back-propagation
    for i in range(epochs):
        network.backprop(mse, x_train, train_output, eta)

    #Test the network on the test data
    print("Test MSE after training network: %.4f" %mse(test_output, network.feed_forward(x_test)))

    #comparing with tensorflow and sklearn
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2
    from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2

    batch_size = len(x_train)
    n_neurons_layer1 = 10
    n_neurons_layer2 = 6
    n_categories = 1

    def create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories, eta):
        model = Sequential()
        model.add(Dense(n_neurons_layer1, activation='sigmoid'))
        model.add(Dense(n_neurons_layer2, activation='sigmoid'))
        model.add(Dense(n_categories, activation=tf.identity))

        sgd = SGD(lr=eta)
        #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])
        model.compile(loss='mean_squared_error')

        return model

    DNN = create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories,eta=eta)
    DNN.fit(x_train, train_output, epochs=epochs, batch_size=batch_size, verbose=0)
    scores = DNN.evaluate(x_test, test_output)

    print("Test MSE from Keras: %.3f" % scores)
