import numpy as np
from CNN_class import CNN_keras
from data_adjustment import extract_data
from sklearn.model_selection import train_test_split
import os
import time as t

paths = ["../images/Apple/",
         "../images/Banana",
         "../images/Kiwi",
         "../images/Mango",
         "../images/Orange",
         "../images/Pear",
         "../images/Tomatoes"]
true_labels = ["apple", "banana", "kiwi", "mango",
               "orange", "pear", "tomato"]


print("Do you want to gray-scale the images [y/n]?")
gray = input()
print()

if gray=="y" or gray=="Y":
    gray = True
elif gray=="n" or gray=="N":
    gray = False
else:
    assert False, "Input '%s' not understood! Try Y/y for gray-scaleing \
or N/n for coloured images." % gray


print("Pleas enter the desired quadratic resolution of data")
print("Must be an integer. If not spesified, 50x50 will be used:")
shape = input()
print()
try:
    im_shape = int(shape)
except:
    im_shape = 50
if gray: data_size = (im_shape, im_shape, 1)
else: data_size = (im_shape, im_shape, 3)



print("Do you want to use a single value for all parameters [single] \
or a set of values for two parameters [set] ?")
params = input()
print()

if params == "single":
    I,J = False,False
elif params == "set":
    print("Which parameters to you want to evalute? Please use the \
notation given below.")
    print("eta = [et]")
    print("lambda = [l]")
    print("epoch = [ep]")
    print("batch sise = [b]")
    print("receptive field = [r]")
    print("neurons in dense layer = [n]")
    print("First parameter:")
    par1 = input()
    print("Second parameter:")
    par2 = input()

    """
    I and J is the index of the two parameters that
    will be evaluated.

    Index   |   Parameter
    ----------------------
    0       |   eta
    1       |   lambda
    2       |   Epochs
    3       |   Batch size
    4       |   Filters
    5       |   Neurons connected
    """

    if par1=="et": I=0
    elif par1=="l": I=1
    elif par1=="ep": I=2
    elif par1=="b": I=3
    elif par1=="r": I=4
    elif par1=="n": I=5
    else: assert False, "'%s' not valid! Please insert a valid input!" % par1

    if par2=="et": J=0
    elif par2=="l": J=1
    elif par2=="ep": J=2
    elif par2=="b": J=3
    elif par2=="r": J=4
    elif par2=="n": J=5
    else: assert False, "'%s' not valid! Please insert a valid input!" % par2

else:
    assert False, "Need to set [single] or [set] as input! Please try again."





eta =  0.01       # optimal parameters found
lmbd = 0.0005     # optimal parameters found
etas = [0.0005,0.001,0.005,0.01]
lmbds = [0.0001,0.0005,0.001,0.005]

epochs = 10     # optimal parameters found
batch_size = 5  # optimal parameters foundx
epochss = [1, 3, 5, 10]
batch_sizes = [1, 3, 5, 10]

rec_field = 3
filters = 20     # optimal parameters found
neuros_con = 100     # optimal parameters found
filterss = [3,10,25,50]
neuros_cons = [10,25,50,100]

"""
params_array contains a list of arrays. The sub-arrays
contains a set of values for different parameters of interest
"""
params_array = [etas, lmbds, epochss, batch_sizes, filterss, neuros_cons]
params_name = ["eta", "lambda", "Epochs", "Batch size", "Kernel size"
                "Neurons connected"]




"""
if im_shape is large (e.g. 200), max_data should be low (e.g. 500)
if im_shape is low (e.g. 50), max_data can be large (e.g. 5000)
(assuming you have only 8 GB of RAM)
"""

#max_data = int(1e5/im_shape)
#max_data = np.min([max_data, 3000])
max_data=500
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
print("Image resolution:         %ix%ix%i" % (im_shape, im_shape, data_size[2]))
print("Max data per run:        ", max_data)
print("Total runs:              ", runs)
print("Total fruits per run:    ", lim_data)
print("---------------------------------------------")
print()





try:
    param1 = params_array[I]
    param2 = params_array[J]
except:
    I,J = 0,1
    param1 = eta
    param2 = lmbd

num_models = len(param1)*len(param2)
CNN_accuracy_train = np.zeros((len(param1), len(param2)))
CNN_accuracy_test = np.zeros((len(param1), len(param2)))
train_acc = np.zeros((len(param1), len(param2)))
test_acc = np.zeros((len(param1), len(param2)))

CNN = []

runs=1

for k in range(runs):
    print("Run:   %i/%i" % (k+1,runs))
    data = extract_data(paths, true_labels, lim_data=lim_data, from_data=k*lim_data)
    data.reshape(im_shape)            # making all data the same shape
    #data.shuffle()             # train_test_split() does the same work
    if gray: data.gray()


    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.hot_vector,
                                                        train_size=0.8,
                                                        test_size=0.2)

    X_train, X_test = X_train/255, X_test/255          # scaling data


    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):

            if I!=False or J!=False:
                print()
                print("%s:          %s" % (params_name[I], p1))
                print("%s:          %s" % (params_name[J], p2))

            if int(k*num_models + i*len(param2)+j) < num_models:

                if params_name[I]==params_name[0]: eta = p1
                elif params_name[I]==params_name[1]: lmbd = p1
                elif params_name[I]==params_name[4]: filters = p1
                elif params_name[I]==params_name[5]: neuros_con = p1

                if params_name[J]==params_name[0]: eta = p2
                elif params_name[J]==params_name[1]: lmbd = p2
                elif params_name[J]==params_name[4]: filters = p2
                elif params_name[J]==params_name[5]: neuros_con = p2


                CNN.append(CNN_keras(input_shape=data_size,
                                     receptive_field=rec_field,
                                     n_filters = filters,
                                     n_neurons_connected = neuros_con,
                                     labels = true_labels,
                                     eta = eta,
                                     lmbd = lmbd))

                if i==0 and j==0: show_model=True
                else: show_model=False
                CNN[int(i*len(param2)+j)].add_layer(show_model=show_model)

            if params_name[I]==params_name[2]: epochs = p1
            elif params_name[I]==params_name[3]: batch_size = p1

            if params_name[J]==params_name[2]: epochs = p2
            elif params_name[J]==params_name[3]: batch_size = p2

            CNN[i*len(param2)+j].model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

            train_acc[i,j] += CNN[i*len(param2)+j].model.evaluate(X_train, y_train, verbose=1)[1]
            test_acc[i,j] += CNN[i*len(param2)+j].model.evaluate(X_test, y_test, verbose=1)[1]
            print("Train accuracy:  ", train_acc[i,j]/(k+1))
            print("Test accuracy:   ", test_acc[i,j]/(k+1))


    data.delete_all_data()          # clear the memory
    print("\n\n")

CNN_accuracy_train = train_acc/runs
CNN_accuracy_test = test_acc/runs


# Save the train and accuracy map for plotting
time_save = t.time()
np.save("../results/plotting_data/acc_train_%s_and_%s_%ix%i_%i.npy" %
       (params_name[I], params_name[J], len(param1), len(param2), time_save),
        CNN_accuracy_train)
np.save("../results/plotting_data/acc_test_%s_and_%s_%ix%i_%i.npy" %
       (params_name[I], params_name[J], len(param1), len(param2), time_save),
        CNN_accuracy_test)
print("Files succesfully saved in the folder '../results/plotting_data/' \
ending with _%i.npy" % time_save)
