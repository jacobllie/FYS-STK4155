import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sb
import sys

from activation_function import sigmoid, identity, relu, leaky_relu,softmax
from data_prep import data_prep
from functions import FrankeFunction
from cost_functions import MSE

class DenseLayer:
    def __init__(self, inputs, outputs, act_func):
        np.random.seed(200)
        #Here we can try with different inializations
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

    def feedforward(self, a):
        for layer in self.layers:
            a = layer(a)
        return a

    def backprop(self, cost, x, y, eta, penalty=0):
        self.feedforward(x)  #using the updated weights and biases to get new output layer
        #Starting with output layer
        L = self.layers
        a = L[-1].a
        delta_l = cost.deriv(y, a)*L[-1].da
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
        L[0].b = L[0].b - eta*delta_l[0, :]

    def backprop2layer(self, cost, x, y, eta, penalty=0):
        self.feedforward(x)  #using the updated weights and biases to get new output layer
        #Starting with output layer
        L = self.layers
        a = L[0].a
        delta_l = cost.deriv(y, a)*L[0].da
        #Updating output layer with the new weights and biases.
        L[0].weights = L[0].weights - eta*(x.T @ delta_l) \
            - eta*penalty*L[0].weights/len(y)
        L[0].b = L[0].b - eta*delta_l[0, :]

    def SGD(self, cost, mini_batch_size, X_train_shuffle, z_train_shuffle, eta, penalty=0):
        for j in range(0,X_train_shuffle.shape[0],mini_batch_size):
            self.backprop(cost, X_train_shuffle[j:j+mini_batch_size],
                z_train_shuffle[j:j+mini_batch_size],eta,penalty)
        return X_train_shuffle,z_train_shuffle


#Test case
if __name__ == "__main__":

    franke_epoch = input("Analyse learning rate vs epochs [Y/n]: ")
    if franke_epoch == "Y" or franke_epoch == "y":
        franke_epoch = True
    elif franke_epoch == "N" or franke_epoch == "n":
        franke_epoch = False
    else:
        print("Please input Y or n!")
        sys.exit()

    franke_lambda = input("Analyse learning rate vs penalty parameter [Y/n]: ")
    if franke_lambda == "Y" or franke_lambda == "y":
        franke_lambda = True
    elif franke_lambda == "N" or franke_lambda == "n":
        franke_lambda = False
    else:
        print("Please input Y or n!")
        sys.exit()

    network_keras = input("Analyse using Keras [Y/n]: ")
    if network_keras == "Y" or  network_keras == "y":
        network_keras = True
    elif network_keras == "N" or network_keras == "n":
        network_keras = False
    else:
        print("Please input Y or n!")
        sys.exit()

    franke_relu = input("Analyse relu activation (learning rate vs epochs) [Y/n]: ")
    if franke_relu == "Y" or franke_relu == "y":
        franke_relu = True
    elif franke_relu == "N" or franke_relu == "n":
        franke_relu = False
    else:
        print("Please input Y or n!")
        sys.exit()

    n = 100
    noise = 0.1
    eta = 0.5
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = np.ravel(FrankeFunction(x, y) + noise*np.random.randn(n, n))

    data = data_prep()
    X = data.X_D(x, y, z, 1)    #X_train,X_test,z_train,z_test
    train_input, test_input, train_output, test_output = data.train_test_split_scale()

    y_train = np.reshape(train_output,(-1, 1))               #the shape was (x,) we needed (x,1) for obvious reasons
    y_test= np.reshape(test_output,(-1, 1))
    x_train = train_input[:, [1, 2]]
    x_test = test_input[:, [1, 2]]



    #Setting up network
    layer1 = DenseLayer(2, 10, sigmoid())
    layer2 = DenseLayer(10, 20, sigmoid())
    layer3 = DenseLayer(20, 1, identity())
    layers = [layer1, layer2, layer3]
    network = NN(layers)
    #Finding MSE on untrained network
    mse = MSE()
    print("Test MSE before training network: %.4f" %mse(y_test, network.feedforward(x_test)))
    #Back-propagation
    m = x_train.shape[0]


    """batch = np.arange(0, m)
    for i in range(500):
        random.shuffle(batch)
        x_train_shuffle = x_train[batch]
        y_train_shuffle = y_train[batch]
        for j in range(0, m, 30):
            #eta = learning_schedule(m*i + j)
            network.backprop(mse, x_train_shuffle[j:j+mini_batch_size],
                y_train_shuffle[j:j+mini_batch_size], eta, penalty=0)

    #Test the network on the test data
    print("Test MSE after training network: %.4f" %mse(y_test, network.feedforward(x_test)))"""

    #comparing with tensorflow and sklearn
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.regularizers import l2
    from keras.optimizers import SGD

    n_neurons_layer1 = 10
    n_neurons_layer2 = 6
    n_categories = 1

    #Finding MSE on untrained network
    #print("Test MSE before training network: %.4f" %mse(test_output, network.feedforward(x_test)))
    penalties = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
    etas = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    epochs = [50, 100, 200, 400, 800, 1000]

    p_len = len(penalties)
    e_len = len(etas)
    ep_len = len(epochs)

    ind = np.arange(0,len(x_train))
    if franke_lambda == True:
        MSE_network = np.zeros((p_len, e_len, 2))      # 2 so we can use plot_acc.py
        for i, penalty in enumerate(penalties):
            for j, eta in enumerate(etas):
                #Setting up network
                layer1 = DenseLayer(2, n_neurons_layer1, sigmoid())
                layer2 = DenseLayer(n_neurons_layer1, n_neurons_layer2, sigmoid())
                layer3 = DenseLayer(n_neurons_layer2, 1, identity())
                layers = [layer1, layer2, layer3]
                network = NN(layers)
                # Back-propagation
                for k in range(100):
                    random.shuffle(ind)
                    x_train_shuffle = x_train[ind]
                    y_train_shuffle = y_train[ind]
                    network.SGD(mse, 100, x_train_shuffle, y_train_shuffle, eta, penalty)
                MSE_network[i,j,0] = mse(y_test, network.feedforward(x_test))
                MSE_network[i,j,1] = mse(y_train_shuffle, network.feedforward(x_train_shuffle))
                progress = int(100*(p_len*(i+1) + (j+1))/(p_len*p_len + e_len))
                print(f"\r Progress: {progress}%", end = "\r")

        plt.figure("FFNN")
        heatmap = sb.heatmap(MSE_network[:,:,0],cmap="viridis_r",
                                      yticklabels=["%.3f" %i for i in penalties],
                                      xticklabels=["%.4f" %j for j in etas],
                                      cbar_kws={'label': 'MSE'},
                                      fmt = ".3",
                                      annot = True,
                                      vmax=0.5)
        plt.yticks(rotation=0)
        heatmap.set_xlabel(r"$\eta$")
        heatmap.set_ylabel(r"$\lambda$")
        heatmap.invert_yaxis()
        heatmap.set_title("MSE on Franke dataset with FFNN")
        fig = heatmap.get_figure()
        fig.savefig("../figures/Franke_FFNN.pdf", bbox_inches='tight',
                                                    pad_inches=0.1)
        plt.show()

    if franke_epoch == True:
        MSE_epoch = np.zeros((ep_len, e_len))      # 2 so we can use plot_acc.py
        for i, eta in enumerate(etas):
            #Setting up network
            layer1 = DenseLayer(2, n_neurons_layer1, sigmoid())
            layer2 = DenseLayer(n_neurons_layer1, n_neurons_layer2, sigmoid())
            layer3 = DenseLayer(n_neurons_layer2, 1, identity())
            layers = [layer1, layer2, layer3]
            network = NN(layers)
            # Back-propagation
            k=0
            for j in range(epochs[-1]+1):
                random.shuffle(ind)
                x_train_shuffle = x_train[ind]
                y_train_shuffle = y_train[ind]
                network.SGD(mse, 100, x_train_shuffle, y_train_shuffle, eta, penalty=0)
                if j in epochs:
                    MSE_epoch[k,i] = mse(y_test, network.feedforward(x_test))
                    progress = int(100*(e_len*(i+1) + (k+1))/(e_len*e_len + ep_len))
                    print(f"\r Progress: {progress}%", end = "\r")
                    k+=1

        plt.figure("EPOCH")
        heatmap = sb.heatmap(MSE_epoch,cmap="viridis_r",
                                      yticklabels=["%d" %i for i in epochs],
                                      xticklabels=["%.4f" %j for j in etas],
                                      cbar_kws={'label': 'MSE'},
                                      fmt = ".3",
                                      annot = True)
        plt.yticks(rotation=0)
        heatmap.set_xlabel(r"$\eta$")
        heatmap.set_ylabel("Epochs")
        heatmap.invert_yaxis()
        heatmap.set_title("MSE on Franke dataset with FFNN")
        fig = heatmap.get_figure()
        plt.show()
        fig.savefig("../figures/Franke_FFNN_epoch.pdf", bbox_inches='tight',
                                                    pad_inches=0.1)

    def create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories, eta, penalty=0):
        model = Sequential()
        model.add(Dense(n_neurons_layer1, activation='sigmoid', kernel_regularizer=l2(penalty)))
        model.add(Dense(n_neurons_layer2, activation='sigmoid', kernel_regularizer=l2(penalty)))
        model.add(Dense(n_categories, activation=tf.identity))

        sgd = SGD(lr=eta)
        #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])
        model.compile(loss='mean_squared_error', optimizer=sgd)
        return model

    if network_keras == True:
        MSE_Keras = np.zeros((len(etas), len(penalties)))      # 2 so we can use plot_acc.py
        for i in range(len(etas)):
            for j in range(len(penalties)):
                DNN = create_neural_network_keras(n_neurons_layer1, n_neurons_layer2,
                    n_categories,eta=etas[i],penalty = penalties[j])
                DNN.fit(x_train, y_train, epochs=100, batch_size=100, verbose=0)
                MSE_Keras[j,i] = DNN.evaluate(x_test, test_output)
                progress = int(100*(e_len*(i+1) + (j+1))/(e_len*e_len + p_len))
                print(f"\r Progress: {progress}%", end = "\r")


        plt.figure("Keras")
        heatmap = sb.heatmap(MSE_Keras,cmap="viridis_r",
                                      yticklabels=["%.3f" %i for i in penalties],
                                      xticklabels=["%.4f" %j for j in etas],
                                      cbar_kws={'label': 'MSE'},
                                      fmt = ".3",
                                      annot = True,
                                      vmax=0.17)
        plt.yticks(rotation=0)
        heatmap.set_xlabel(r"$\eta$")
        heatmap.set_ylabel(r"$\lambda$")
        heatmap.invert_yaxis()
        heatmap.set_title("MSE on Franke dataset with Keras")
        fig = heatmap.get_figure()
        plt.show()
        fig.savefig("../figures/Franke_Keras.pdf", bbox_inches='tight',
                                                    pad_inches=0.1)

    if franke_relu == True:
        MSE_relu = np.zeros((len(etas), len(epochs)))      # 2 so we can use plot_acc.py
        for i, eta in enumerate(etas):
            #Setting up network
            layer1 = DenseLayer(2, n_neurons_layer1, relu())
            layer2 = DenseLayer(n_neurons_layer1, n_neurons_layer2, relu())
            layer3 = DenseLayer(n_neurons_layer2, 1, identity())
            layers = [layer1, layer2, layer3]
            network = NN(layers)
            k=0
            for j in range(epochs[-1]+1):
                random.shuffle(ind)
                x_train_shuffle = x_train[ind]
                y_train_shuffle = y_train[ind]
                network.SGD(mse, 100, x_train_shuffle, y_train_shuffle, eta, penalty=0)
                if j in epochs:
                    MSE_relu[k,i] = mse(y_test, network.feedforward(x_test))
                    progress = int(100*(e_len*(i+1) + (k+1))/(e_len*e_len + ep_len))
                    print(f"\r Progress: {progress}%", end = "\r")
                    k+=1

        plt.figure("relu")
        heatmap = sb.heatmap(MSE_relu,cmap="viridis_r",
                                      yticklabels=["%d" %i for i in epochs],
                                      xticklabels=["%.4f" %j for j in etas],
                                      cbar_kws={'label': 'MSE'},
                                      fmt = ".3",
                                      annot = True)
        plt.yticks(rotation=0)
        heatmap.set_xlabel(r"$\eta$")
        heatmap.set_ylabel("Epochs")
        heatmap.invert_yaxis()
        heatmap.set_title("MSE on Franke dataset with FFNN using relu")
        fig = heatmap.get_figure()
        fig.savefig("../figures/Franke_FFNN_relu.pdf", bbox_inches='tight',
                                                    pad_inches=0.1)
        plt.show()
