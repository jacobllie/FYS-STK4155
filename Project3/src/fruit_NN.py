import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import Network as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from activation_function import sigmoid, softmax, relu
from extract_fruit import ExtractData
from cost_functions import CE, accuracy
from numpy import random
import seaborn as sb
path = "../images"

num_fruits = 2

apples = ExtractData(path, "Apple", 500)
apples.gray_scale()
Apple = apples()
apple_labels = np.array(["Apple" for i in range(len(Apple))])


bananas = ExtractData(path, "Banana",500)
bananas.gray_scale()
Banana = bananas()
banana_labels = np.array(["Banana" for i in range(len(Banana))])

fruit_list = np.concatenate([Apple, Banana])
fruit_label = np.concatenate([apple_labels,banana_labels])
labels = np.array(["Apple","Banana"])

num_label = np.zeros(len(fruit_label))
num_label[:] = fruit_label == "Banana"

one_hot = np.zeros((len(fruit_label), num_fruits))
for i in range(num_fruits):
    one_hot[:, i] = fruit_label  == labels[i]

#splitting and scaling data
train_fruit, test_fruit, train_label, test_label,one_hot_train,_\
    = train_test_split(fruit_list, num_label,one_hot,test_size = 0.3)
scaler = StandardScaler()
scaler.fit(train_fruit)
train_fruit = scaler.transform(train_fruit)
test_fruit = scaler.transform(test_fruit)




layer1 = nn.DenseLayer(fruit_list.shape[1],100,sigmoid(),Glorot = True)
#first layer will take all width and height pixels as input
layer2 = nn.DenseLayer(100,50,sigmoid(),Glorot = True)
layer3 = nn.DenseLayer(50,num_fruits,softmax(),Glorot = True)
layers = [layer1,layer2,layer3]

ce = CE()
m = train_fruit.shape[0]
batch = np.arange(0,m)
epochs = 200
cost = np.zeros((epochs,2))   #train is first column test is second
penalties = np.logspace(-4,-1,4)
etas = np.logspace(-3,0,4)
network = nn.NN(layers,ce)
accuracy_map = np.zeros((len(penalties),len(etas),2))

for i in range(len(penalties)):
    for j in range(len(etas)):
        for k in range(epochs):
            random.shuffle(batch)
            train_fruit_shuffle = train_fruit[batch]
            one_hot_shuffle = one_hot_train[batch]
            train_label_shuffle = train_label[batch]
            network.SGD(100, train_fruit_shuffle, one_hot_shuffle, etas[j],penalties[i])
            fruit_pred_train = np.argmax(network.feedforward(train_fruit_shuffle),axis=1)
            fruit_pred_test = np.argmax(network.feedforward(test_fruit),axis = 1)
            #cost[k,0] = accuracy()(train_label_shuffle,fruit_pred_train)
            #cost[k,1] = accuracy()(test_label,fruit_pred_test)
            accuracy_map[i,j,0] = accuracy()(train_label_shuffle,fruit_pred_train)
            accuracy_map[i,j,1] = accuracy()(test_label,fruit_pred_test)

heatmap = sb.heatmap(accuracy_map[:,:,1],cmap="viridis",
                              xticklabels=etas,
                              yticklabels=penalties,
                              cbar_kws={'label': 'Accuracy'},
                              fmt = ".3",
                              annot = True)
plt.yticks(rotation=0)
heatmap.set_xlabel(r"$\eta$")
heatmap.set_ylabel(r"$\lambda$")
heatmap.invert_yaxis()
heatmap.set_title("Accuracy on pictures of fruit")
fig = heatmap.get_figure()
plt.show()



"""
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.plot(cost[:,0],label = "Train accuracy")
plt.plot(cost[:,1],label= "Test accuracy")
plt.legend()
plt.show()
"""
