from numpy import random
from activation_function import sigmoid, identity, relu, leaky_relu
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from data_prep import data_prep
from functions import FrankeFunction

random.seed(100)

class dense_layer:
    def __init__(self, inputs, outputs, act_func):
        self.weights = random.randn(inputs, outputs)
        self.b = random.randn(1, outputs)
        self.act_func = act_func

    def __call__(self, X):
        #calculate activation
        #print(self.b.shape)
        #print(X.shape)
        #print(self.weights.shape)

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

    def backprop(self, cost, x, y, eta):
        #Starting with output layer
        L = self.layers
        a = L[-1].a
        delta_l = cost.deriv(y, a)*L[-1].da
        #Looping over layer from output to 1st hidden
        for i in reversed(range(1, len(L)-1)):
            #print(L[i+1].weights.T.shape)
            #print(delta_l.shape)
            delta_l = (delta_l@L[i+1].weights.T)*L[i].da
            #Updating weights
            L[i].weights = L[i].weights - eta*(L[i-1].a.T@delta_l)

            #Updating biases
            L[i].b = L[i].b - eta*delta_l[0,:]
        #First hidden layer, here a = x
        delta_l = (delta_l@L[1].weights.T)*L[0].da
        L[0].weights = L[0].weights - eta*(x.T@delta_l)
        L[0].b = L[0].b - eta*delta_l[0,:]



        self.feed_forward(x)

class MSE:
    def __call__(self, y, a):
        return 1/len(y)*np.sum(a - y)**2

    def deriv(self, y, a):
        return 2/len(y)*(a - y)

#Test case
if __name__ == "__main__":

    n = 100
    noise = 0.1
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)
    x,y = np.meshgrid(x,y)
    z = np.ravel(FrankeFunction(x,y) + noise*np.random.randn(n,n))

    data = data_prep(x,y,z,1)
    train_input, test_input, train_output, test_output = data()    #X_train,X_test,z_train,z_test
    train_output = np.reshape(train_output,(-1,1))               #the shape was (x,) we needed (x,1) for obvious reasons
    test_output = np.reshape(test_output,(-1,1))
    x_train = train_input[:, [1,2]]
    x_test = test_input[:, [1,2]]

    layer1 = dense_layer(2, 10, sigmoid())
    layer2 = dense_layer(10, 6, sigmoid())
    layer3 = dense_layer(6, 1, identity())

    layers = [layer1, layer2, layer3]
    network = NN(layers)
    network.feed_forward(x_train)  #sending in x and y from design matrix.
    mse = MSE()
    for i in range(1000):
        network.backprop(MSE(), x_train,train_output, 0.1)

    #Test the network on the test data
    print(mse(train_output, layers[-1].a))
    network.feed_forward(x_test)
    print(mse(test_output, layers[-1].a))

    network.feed_forward(x_test)
    #print(mse(test_output,layers[-1].a))


    #comparing with tensorflow and sklearn

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2
    from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2
    from keras.optimizers import SGD

    epochs = 100
    batch_size = n
    n_neurons_layer1 = 10
    n_neurons_layer2 = 5
    n_categories = 1
    eta_vals = np.array([0.5])
    lmbd_vals = np.array([0])


    def create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories, eta, lmbd):
        model = Sequential()
        model.add(Dense(n_neurons_layer1, activation='sigmoid'))
        model.add(Dense(n_neurons_layer2, activation='sigmoid'))
        model.add(Dense(n_categories))

        sgd = SGD(lr=eta)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])

        return model

    DNN_keras = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)










    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            DNN = create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories,eta=eta, lmbd=lmbd)
            DNN.fit(x_train, train_output, epochs=epochs, batch_size=batch_size, verbose=0)
            scores = DNN.evaluate(x_train, train_output)

            DNN_keras[i][j] = DNN

            print("Learning rate = ", eta)
            print("Lambda = ", lmbd)
            print("Test mse: %.3f" % scores[1])
            print()
