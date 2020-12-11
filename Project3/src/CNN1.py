import numpy as np
import matplotlib.pyplot as plt
from data_adjustment import extract_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers, regularizers,initializers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sb
from numpy import save
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
import time

class CNN_keras:
    def __init__(self, input_shape, receptive_field, n_filters,
                                    n_neurons_connected, labels, eta, lmbd):
        """
        input_shape: the shape of input data
        receptive_field: small area connected to each neuron in next layer (size of filters I think!?!?!?!?!)
        n_filters: number of filters that is multiplied to the input image
        n_neurons_connected:
        labels:
        """

        self.inp = input_shape
        self.recf = receptive_field
        self.nfilt = n_filters
        self.nneur = n_neurons_connected
        self.categories = len(np.unique(labels))
        self.eta = eta
        self.lmbd = lmbd

        self.model = Sequential()


    def add_layer(self, show_model=False):
        """
        First convolutional layer must contain the input shape,
        the other layers is not dependent of it
        """
        self.model.add(Conv2D(20, (self.recf,self.recf),
            input_shape=self.inp,activation='relu', padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(20, (self.recf,self.recf),
            input_shape=self.inp,activation='relu', padding = 'same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.model.add(Conv2D(2*self.nfilt, (self.recf,self.recf)))
        #self.model.add(MaxPooling2D(pool_size=(2,2)))


        """
        Before we can add dense layers, the output from previous
        3D layers must be flatten (i.e. convert to vector form)
        """
        self.model.add(Flatten())
        #self.model.add(Dense(2*self.nneur, activation="relu",
        #                     kernel_regularizer=regularizers.l2(self.lmbd)))
        self.model.add(Dense(self.nneur, activation="relu",
                             kernel_regularizer=regularizers.l2(self.lmbd)))
        self.model.add(Dense(self.categories, activation="softmax",
                             kernel_regularizer=regularizers.l2(self.lmbd)))

        sgd = optimizers.SGD(lr=self.eta)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd, metrics=['accuracy'])

        if show_model==True: self.model.summary()


if __name__ == '__main__':

    paths = ["../images/Apple/Total Number of Apples",
             "../images/Banana",
             "../images/Kiwi/Total Number of Kiwi fruit",
             "../images/Mango",
             "../images/Orange",
             "../images/Pear",
             "../images/Tomato"]
    true_labels = ["apple", "banana", "kiwi", "mango",
                   "orange", "pear", "tomato"]


    gray = True

    im_shape = 100
    #data_size = data.data[0].shape
    if gray == False:
        data_size = (im_shape, im_shape, 3)
    else:
        data_size = (im_shape,im_shape,1)
    rec_field = 3
    filters = 20
    neuros_con = 100

    eta = [0.01]#[0.0005,0.001,0.005,0.01]
    lmbd = [0.0005]#[0.0001,0.0005,0.001,0.005]
    epochs = 10
    batch_size = 5

    """
    if im_shape is large (e.g. 200), max_data should be low (e.g. 500)
    if im_shape is low (e.g. 50), max_data can be large (e.g. 5000)
    """
    max_data = 3000
    lim_data = int(max_data/len(paths))
    #tot_data = 0
    lens = []
    for path in paths:
        #tot_data += len(os.listdir("./"+path))
        lens.append(len(os.listdir("./"+path)))
    min_len = np.min(lens)
    tot_data = (min_len * len(paths))
    print(tot_data)
    runs = int(tot_data / max_data)    #we will use the last run to test the final data

    print("---------------------------------------------")
    print("Unique fruits:           ", len(paths))
    print("Total fruits:            ", tot_data)
    print("Image resolution:         %ix%i" % (im_shape, im_shape))
    print("Max data set to:         ", max_data)
    print("Total runs:              ", runs)
    print("Total fruit per run:     ", lim_data)
    print("---------------------------------------------")
    print()
    pred = []
    pred_labels = []
    num_label = []
    score_train = np.zeros((runs-1,2))
    score_val = np.zeros((runs-1,2))
    accuracy_map = np.zeros((len(eta),len(lmbd)))
    frac_data = np.zeros(runs-1)
    accuracy_epoch = np.zeros((runs-1,epochs))

    best_score = 0
    start_time = time.time()
    for j in range(len(eta)):
        for k in range(len(lmbd)):
            CNN = CNN_keras(input_shape=data_size,
                            receptive_field=rec_field,
                            n_filters = filters,
                            n_neurons_connected = neuros_con,
                            labels = true_labels,
                            eta = eta[j],
                            lmbd = lmbd[k])

            CNN.add_layer(show_model=True)

            for i in range(runs-1):
                frac_data[i] = (i+1)*lim_data     #images trained
                print("Run:   ", i+1)
                data = extract_data(paths, true_labels,
                    lim_data=lim_data, from_data=i*lim_data)
                data.reshape(im_shape)
                if gray: data.gray()



                #del data.data
                X_train, X_val, y_train, y_val = train_test_split(data.data,
                                                                 data.hot_vector,
                                                                 train_size=0.8)
                print("{} images used for validation".format(X_val.shape[0]))
                X_train,X_val = X_train/255,X_val/255  #Scaling each pixel value
                #with max RGB value

                history = CNN.model.fit(X_train, y_train, epochs=epochs,
                             batch_size=batch_size, verbose=1)
                accuracy_epoch[i] = history.history["accuracy"]
                print(accuracy_epoch[i])
                score_train[i] = CNN.model.evaluate(X_train,y_train)
                score_val[i] = CNN.model.evaluate(X_val, y_val, verbose=1)
                data.delete_all_data()          # clear the memory
                print("Accuracy on validation data with eta = {:.4}, lambda\
                      = {:.4} is {:.4}".format(eta[j],lmbd[k],score_val[i,1]))
            trainig_time = time.time()-start_time
            #running test data through the network
            test_data = extract_data(paths,true_labels,lim_data=lim_data,
                                    from_data=runs*lim_data)
            test_data.reshape(im_shape)

            if gray: test_data.gray()
            scaled_test = test_data.data/255
            scores = CNN.model.evaluate(scaled_test,test_data.hot_vector)
            pred = CNN.model.predict(scaled_test)
            accuracy_map[j][k] = score_val[i,1]
            print("Accuracy on holy test data for lambda = {:.4},\
                 eta = {:.4} is {:.4}".format(lmbd[k],eta[j],accuracy_map[j][k]))
    #making confusion matrix
            if score_val[-1,1] > best_score:
                pred_num_label = np.argmax(pred,axis=1)
                true_num_label = np.argmax(test_data.hot_vector,axis=1)
                conf_matrix = confusion_matrix(true_num_label,pred_num_label,normalize="true")
                best_score = scores[1]
    #storing the values necessary for plotting
                np.save("../data/CNN_confusion_matrix_100_gray",conf_matrix)
    #test_im = np.array(Image.open("./test_images/test_apple.png"))
    #np.save("../data/CNN_accuracy_map_100nneur",accuracy_map)
    #np.save("../data/CNN_accuracy_validation",score_val)
    #np.save("../data/CNN_frac_data",frac_data)
    #np.save("../data/CNN_accuracy_epoch",accuracy_epoch)
    print("Time spent on training {:.4}s".format(trainig_time))

    """
    Image configuration example
    """

    """
    test_path = ["/test_images"]
    test_label = ["apple"]

    test = extract_data(test_path, test_label, copy_data=True)



    test.reshape(50)


    for layer in CNN.model.layers:
	# check for convolutional layer
	   if 'conv' not in layer.name:
	        continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    plt.subplot(1,4,1)
    plt.imshow(filters[:,:,:,0])
    plt.subplot(1,4,2)
    plt.imshow(filters[:,:,:,1])
    plt.subplot(1,4,3)
    plt.imshow(filters[:,:,:,2])
    plt.subplot(1,4,4)
    plt.imshow(filters[:,:,:,3])
    gray = False
    plt.figure("Reshape", figsize=(5,4))


    if gray:
        test.gray()
        plt.imshow(test.data[0,...,0],cmap="gray")
        save_name = "Reshape_gray_example.pdf"
    else:
        plt.imshow(test.data[0])
        save_name = "Reshape_example.pdf"
    plt.title("Reshaped image (%ix%i)"%(50,50))

    plt.savefig(save_name)

    plt.tight_layout()
    plt.show()
    """
