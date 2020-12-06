import numpy as np
import matplotlib.pyplot as plt
from CNN import CNN_keras
from data_adjustment import extract_data
import seaborn as sb
from sklearn.model_selection import train_test_split
import os


paths = ["/data_images/Mango",
         "/data_images/Tomato"]
true_labels = ["mango", "tomato"]

"""
data = extract_data(paths, true_labels, lim_data=2000)
im_shape = 50
data.reshape(im_shape)
data.shuffle()
#data.gray()



X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                    data.hot_vector,
                                                    train_size=0.8)
X_train, X_test = X_train/255, X_test/255
"""


gray = False

im_shape = 30
#data_size = data.data[0].shape
if gray: data_size = (im_shape, im_shape, 1)
else: data_size = (im_shape, im_shape, 3)

rec_field = 3
filters = 20
#filters = np.linspace(10,50,7).astype("int")
neuros_con = 50
#neuros_con = np.linspace(30,100,7).astype("int")

#eta = 0.0001
etas = np.logspace(-5,-1,3)
#lmbd = 0.001
lmbds = np.logspace(-5,-1,3)

epochs = 3
#epochss = np.linspace(10,150,7).astype("int")
batch_size = 10
#batch_sizes = np.linspace(1,50,7).astype("int")

"""
if im_shape is large (e.g. 200), max_data should be low (e.g. 500)
if im_shape is low (e.g. 50), max_data can be large (e.g. 5000)
"""
max_data = 5000
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

CNN_accuracy_train = np.zeros((len(etas), len(lmbds)))
CNN_accuracy_test = np.zeros((len(etas), len(lmbds)))




for i, eta in enumerate(etas):
    for j, lmbd in enumerate(lmbds):
        print()
        print("eta:     ", eta)
        print("lambda:  ", lmbd)

        CNN = CNN_keras(input_shape=data_size,
                        receptive_field=rec_field,
                        n_filters = filters,
                        n_neurons_connected = neuros_con,
                        labels = true_labels,
                        eta = eta,
                        lmbd = lmbd)

        CNN.add_layer(show_model=False)

        #runs = 1

        train_acc = 0
        test_acc = 0

        for i in range(runs):
            print("Run:   %i/%i" % (i+1,runs))
            data = extract_data(paths, true_labels, lim_data=lim_data, from_data=i*lim_data)
            data.reshape(im_shape)            # making all data the same shape
            data.shuffle()
            if gray: data.gray()

            X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                                data.hot_vector,
                                                                train_size=0.8,
                                                                test_size=0.2)

            X_train, X_test = X_train/255, X_test/255          # scaling data



            CNN.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            #scores = CNN.model.evaluate(X_test, y_test, verbose=1)

            train_acc += CNN.model.evaluate(X_train, y_train, verbose=1)[1]
            test_acc += CNN.model.evaluate(X_test, y_test, verbose=1)[1]

            data.delete_all_data()          # clear the memory

        CNN_accuracy_train[i,j] = train_acc/runs
        CNN_accuracy_test[i,j] = test_acc/runs
        print("Train accuracy:  ", CNN_accuracy_train[i,j])
        print("Test accuracy:   ", CNN_accuracy_test[i,j])
        print("\n\n")



plt.close()
fig, ax = plt.subplots(figsize = (10, 10))
sb.heatmap(CNN_accuracy_train, annot=True, ax=ax, cmap="viridis", vmax=1,
                            xticklabels=["%.0e" %i for i in lmbds],
                            yticklabels=["%.0e" %i for i in np.flip(etas)],
                            cbar_kws={'label': 'Accuracy'})
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")


fig, ax = plt.subplots(figsize = (10, 10))
sb.heatmap(CNN_accuracy_test, annot=True, ax=ax, cmap="viridis", vmax=1,
                           xticklabels=["%.0e" %i for i in lmbds],
                           yticklabels=["%.0e" %i for i in np.flip(etas)],
                           cbar_kws={'label': 'Accuracy'})
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()





#
