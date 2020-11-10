import numpy as np
import matplotlib.pyplot as plt

from numpy import random
from sklearn import datasets

from activation_function import sigmoid, softmax, identity
from functions import FrankeFunction
from data_prep import data_prep
from NN import dense_layer, NN
from cost_functions import accuracy, CE, MSE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sb
# download MNIST dataset
data = datasets.load_digits()
labels = data.target.reshape(-1,1)
N = labels.shape[0]
inputs = data.images.reshape(N,-1)
#inputs = inputs[:N-500]
#labels = labels[:N-500]
#N = N-500
features = inputs.shape[1]

#prep data
ind = np.arange(0,N)
random.shuffle(ind)
X = inputs[ind]         #input
Y = labels[ind]         #output


data = data_prep()
data.MNIST(data,X,Y)
X_train,X_test,Y_train,Y_test = data.train_test_split_scale()

Y_test = Y_test.reshape(len(Y_test))

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#set up one-hot vector
one_hot = np.zeros((Y_train.shape[0], 10))
for i in range(Y_train.shape[0]):
    one_hot[i,Y_train[i]] = 1

#Make neural network with zero hidden layer
output_layer = dense_layer(features, 10, softmax())

layers = [output_layer]
log_net = NN(layers)
cost_func = CE()    #using cross-entropy as cost function

epochs = 500
mini_batch_size = N//200
eta = np.logspace(-4,-1,4)
penalty = np.logspace(-5,0,4)
m = X_train.shape[0]

ind = np.arange(0, X_train.shape[0])
cost_array = np.zeros((len(eta),len(penalty)))
cost_best = 2
for i in range(len(eta)):
    for j in range(len(penalty)):
        for k in range(epochs):     #looping epochs
            random.shuffle(ind)
            X_train = X_train[ind]
            one_hot = one_hot[ind]
            for l in range(0,m,mini_batch_size):
                log_net.backprop2layer(cost_func, X_train[l:l+mini_batch_size],
                    one_hot[l:l+mini_batch_size], eta[i],penalty[j])
            Y_pred = np.argmax(log_net.feed_forward(X_test), axis=1)
            #print(Y_pred)
            #print(Y_test)

            cost_array[i][j] = accuracy()(Y_test, Y_pred)*100

            if cost_array[i][j] > cost_best:
                print(cost_array[i][j])
                penalty_best = penalty[j]
                eta_best = eta[i]
                cost_best = cost_array[i][j]
                Y_pred_best = Y_pred
                Y_test_best = Y_test



numbers = np.arange(0,10)
confusion_matrix = confusion_matrix(Y_pred_best,Y_test_best,normalize="true")
heatmap = sb.heatmap(confusion_matrix,cmap="viridis",
                              xticklabels=["%d" %i for i in numbers],
                              yticklabels=["%d" %i for i in numbers],
                              cbar_kws={'label': 'Accuracy'},
                              fmt = ".5",
                              edgecolor="none")
heatmap.set_xlabel("pred")
heatmap.set_ylabel("true")

heatmap.set_title(r"MNIST prediction accuracy with $\lambda$ = {:.1e} $\eta$ = {:.1e}"\
        .format(penalty_best,eta_best))
fig = heatmap.get_figure()
fig.savefig("MNIST_confusion.pdf",bbox_inches='tight',
                                  pad_inches=0.1,
                                  dpi = 1200)
plt.show()
