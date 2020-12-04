import numpy as np
import matplotlib.pyplot as plt
from CNN import CNN_keras
from data_adjustment import extract_data
import seaborn as sb
from sklearn.model_selection import train_test_split


paths = ["/data_images/Banana",
         "/data_images/Tomato"]
true_labels = ["banana", "tomato"]


data = extract_data(paths, true_labels, lim_data=2000)
im_shape = 50
data.reshape(im_shape)
data.shuffle()
#data.gray()



X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                    data.hot_vector,
                                                    train_size=0.8)
X_train, X_test = X_train/255, X_test/255

data_size = data.data[0].shape
rec_field = 3
filters = 20
#filters = np.linspace(10,50,7).astype("int")
neuros_con = 50
#neuros_con = np.linspace(30,100,7).astype("int")

#eta = 0.0001
etas = np.logspace(-5,-1,5)
#lmbd = 0.001
lmbds = np.logspace(-5,-1,5)

epochs = 1
#epochss = np.linspace(10,150,7).astype("int")
batch_size = 10
#batch_sizes = np.linspace(1,50,7).astype("int")

CNN_score_train = np.zeros((len(etas), len(lmbds)))
CNN_score_test = np.zeros((len(etas), len(lmbds)))

for i, eta in enumerate(etas):
    for j, lmbd in enumerate(lmbds):
        print()
        print("eta: ", eta)
        print("lambda: ", lmbd)

        CNN = CNN_keras(input_shape=data_size,
                        receptive_field=rec_field,
                        n_filters = filters,
                        n_neurons_connected = neuros_con,
                        labels = true_labels,
                        eta = eta,
                        lmbd = lmbd)

        CNN.add_layer(show_model=False)

        CNN.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        train_scores = CNN.model.evaluate(X_train, y_train, verbose=1)
        test_scores = CNN.model.evaluate(X_test, y_test, verbose=1)

        CNN_score_train[i,j] = train_scores[1]
        CNN_score_test[i,j] = test_scores[1]




fig, ax = plt.subplots(figsize = (10, 10))
sb.heatmap(CNN_score_train, annot=True, ax=ax, cmap="viridis", vmax=1,
                            xticklabels=["%.0e" %i for i in lmbds],
                            yticklabels=["%.0e" %i for i in etas],
                            cbar_kws={'label': 'Accuracy'})
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
#plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sb.heatmap(CNN_score_test, annot=True, ax=ax, cmap="viridis", vmax=1,
                           xticklabels=["%.0e" %i for i in lmbds],
                           yticklabels=["%.0e" %i for i in etas],
                           cbar_kws={'label': 'Accuracy'})
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()





#
