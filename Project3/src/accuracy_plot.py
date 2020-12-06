import numpy as np
import matplotlib.pyplot as plt
from CNN import CNN_keras
from data_adjustment import extract_data
import seaborn as sb
from sklearn.model_selection import train_test_split
import os

paths = ["/data_images/Apple",
         "/data_images/Banana",
         "/data_images/Kiwi",
         "/data_images/Mango",
         "/data_images/Orange",
         "/data_images/Pear",
         "/data_images/Tomato"]
true_labels = ["apple", "banana", "kiwi", "mango",
               "orange", "pear", "tomato"]

"""
paths = ["/data_images/Mango",
         "/data_images/Tomato"]
true_labels = ["mango", "tomato"]
"""


gray = False

im_shape = 30
#data_size = data.data[0].shape
if gray: data_size = (im_shape, im_shape, 1)
else: data_size = (im_shape, im_shape, 3)

length_of_params = 5

rec_field = 3
filters = 20
filterss = np.linspace(10,50,length_of_params).astype("int")
neuros_con = 50
neuros_cons = np.linspace(30,100,length_of_params).astype("int")

eta = 0.0001
etas = np.logspace(-5,-1,length_of_params)
lmbd = 0.001
lmbds = np.logspace(-5,-1,length_of_params)

epochs = 3
epochss = np.linspace(5,30,length_of_params).astype("int")
batch_size = 10
batch_sizes = np.linspace(1,30,length_of_params).astype("int")


"""
params_array contains a list of arrays. The arrays contains increasing
values of different parameters
"""

params_array = [filterss, neuros_cons, etas, lmbds, epochss, batch_sizes]
params_name = ["Filters", "Neurons connected", "eta", "lambda",
               "Epochs", "Batch size"]

"""
if im_shape is large (e.g. 200), max_data should be low (e.g. 500)
if im_shape is low (e.g. 50), max_data can be large (e.g. 5000)
"""
max_data = 2500
lim_data = int(max_data/len(paths))
lens = []
for path in paths:
    lens.append(len(os.listdir("./"+path)))
min_len = np.min(lens)
tot_data = min_len * len(paths)
runs = int(tot_data / max_data) + 1

print("---------------------------------------------")
print("Unique fruits:           ", len(paths))
print("Total fruits:            ", tot_data)
print("Image resolution:         %ix%ix%i" % (data_size[0], data_size[1], data_size[2]))
print("Max data set to:         ", max_data)
print("Total runs:              ", runs)
print("Total fruit per run:     ", lim_data)
print("---------------------------------------------")
print()


"""
I and J is the index of the two parameters that
will be evaluated.

Index   |   Parameter
----------------------
0       |   Filters
1       |   Neurons connected
2       |   eta
3       |   lambda
4       |   Epochs
5       |   Batch size
"""

I = 2
J = 3

param1 = params_array[I]
param2 = params_array[J]

num_models = len(param1)*len(param2)
CNN_accuracy_train = np.zeros((len(param1), len(param2)))
CNN_accuracy_test = np.zeros((len(param1), len(param2)))
train_acc = np.zeros((len(param1), len(param2)))
test_acc = np.zeros((len(param1), len(param2)))

CNN = []
#runs = 1


for k in range(runs):
    print("Run:   %i/%i" % (k+1,runs))
    data = extract_data(paths, true_labels, lim_data=lim_data, from_data=k*lim_data)
    data.reshape(im_shape)            # making all data the same shape
    #data.shuffle()
    if gray: data.gray()

    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.hot_vector,
                                                        train_size=0.8,
                                                        test_size=0.2)

    X_train, X_test = X_train/255, X_test/255          # scaling data


    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):
            print()
            print("%s:          %s" % (params_name[I], p1))
            print("%s:          %s" % (params_name[J], p2))

            if int(k*num_models + i*len(param2)+j) < num_models:

                if params_name[I]==params_name[0]: filters = p1
                elif params_name[I]==params_name[1]: neuros_con = p1
                elif params_name[I]==params_name[2]: eta = p1
                elif params_name[I]==params_name[3]: lmbd = p1

                if params_name[J]==params_name[0]: filters = p2
                elif params_name[J]==params_name[1]: neuros_con = p2
                elif params_name[J]==params_name[2]: eta = p2
                elif params_name[J]==params_name[3]: lmbd = p2


                ["Filters", "Neurons connected", "eta", "lambda",
                               "Epochs", "Batch size"]

                CNN.append(CNN_keras(input_shape=data_size,
                                     receptive_field=rec_field,
                                     n_filters = filters,
                                     n_neurons_connected = neuros_con,
                                     labels = true_labels,
                                     eta = eta,
                                     lmbd = lmbd))
                CNN[int(i*len(param2)+j)].add_layer(show_model=False)

            if params_name[I]==params_name[4]: epochs = p1
            elif params_name[I]==params_name[5]: batch_size = p1

            if params_name[J]==params_name[4]: epochs = p2
            elif params_name[J]==params_name[5]: batch_size = p2

            CNN[i*len(param2)+j].model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            #scores = CNN.model.evaluate(X_test, y_test, verbose=1)

            train_acc[i,j] += CNN[i*len(param2)+j].model.evaluate(X_train, y_train, verbose=1)[1]
            test_acc[i,j] += CNN[i*len(param2)+j].model.evaluate(X_test, y_test, verbose=1)[1]
            print("Train accuracy:  ", train_acc[i,j]/(k+1))
            print("Test accuracy:   ", test_acc[i,j]/(k+1))


    data.delete_all_data()          # clear the memory
    print("\n\n")

CNN_accuracy_train = train_acc/runs
CNN_accuracy_test = test_acc/runs



plt.close()
fig, ax = plt.subplots(figsize = (6, 6))
sb.heatmap(CNN_accuracy_train, annot=True, ax=ax, cmap="viridis", vmax=1,
                            xticklabels=["%s" %i for i in param2],
                            yticklabels=["%s" %i for i in param1],
                            cbar_kws={'label': 'Accuracy'})
ax.set_title("Training Accuracy")
ax.set_ylabel("%s" % params_name[I])
ax.set_xlabel("%s" % params_name[J])


fig, ax = plt.subplots(figsize = (6, 6))
sb.heatmap(CNN_accuracy_test, annot=True, ax=ax, cmap="viridis", vmax=1,
                           xticklabels=["%s" %i for i in param2],
                           yticklabels=["%s" %i for i in param1],
                           cbar_kws={'label': 'Accuracy'})
ax.set_title("Test Accuracy")
ax.set_ylabel("%s" % params_name[I])
ax.set_xlabel("%s" % params_name[J])
plt.show()





#
