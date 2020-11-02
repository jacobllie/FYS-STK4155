import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from activation_function import sigmoid,softmax,identity
from functions import FrankeFunction
from data_prep import data_prep
from NN import dense_layer,NN,MSE
from activation_function import sigmoid,softmax,identity
from cost_functions import accuracy

# download MNIST dataset
data = datasets.load_digits()
inputs = data.images
labels = data.target

cost = accuracy()


instance = data_prep()
instance.MNIST(data,inputs,labels)

train_input,test_input,train_output,test_output = instance.train_test_split_scale()

train_output = np.reshape(train_output,(-1,1))               #the shape was (x,) we needed (x,1) for obvious reasons
test_output = np.reshape(test_output,(-1,1))
x_train = train_input[:, [1,2]]
x_test = test_input[:, [1,2]]

layer1 = dense_layer(2, 10, sigmoid())
layer2 = dense_layer(10, 6, sigmoid())
layer3 = dense_layer(6, 1, softmax())



layers = [layer1, layer2, layer3]
network = NN(layers)
network.feed_forward(x_train)  #sending in x and y from design matrix.
mse = MSE()
print(test_output, network.feed_forward(test_output))
for i in range(1000):
    network.backprop(mse, x_train,train_output, 0.5)

#Test the network on the test data
print(mse(test_output, network.feed_forward(test_output)))

# choose some random images to display
indices = np.arange(len(inputs))
random_indices = np.random.choice(indices, size=5)

for i, image in enumerate(data.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % data.target[random_indices[i]])
plt.show()
