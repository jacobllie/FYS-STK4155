from CNN_class import CNN_keras
from data_adjustment import extract_data
import sys
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image
from tensorflow.keras.models import Model
paths = ["../images/Apple",
           "../images/Banana",
           "../images/Kiwi",
           "../images/Mango",
           "../images/Orange",
           "../images/Pear",
           "../images/Tomatoes"]
true_labels = ["apple", "banana", "kiwi", "mango",
             "orange", "pear", "tomato"]

"""
The first lines of the code represent user interface,
and let's the user decide certain parameters.
"""
gray = input("gray/color?: ")
if gray == "gray":
  gray = True
elif gray == "color":
  gray = False
else:
  print("Please input gray or color!")
  sys.exit()

print("You can either run for optimal learning rate and penalty. \n\
      Or you can run for different learning rates and penalty")
eta_lambda = input("Write [optimal/different] ")
if eta_lambda == "optimal":
  eta_lambda = True
elif eta_lambda == "different":
  eta_lambda = False
else:
  print("Please input optimal or different!")
  sys.exit()

create_confusion = input("Save Confusion matrix [Y/n] ")
if create_confusion == "Y" or create_confusion == "y":
  create_confusion = True
elif create_confusion == "N" or create_confusion == "n":
  create_confusion = False
else:
  print("Please input Y or n!")
  sys.exit()

if eta_lambda == False:
    create_accuracy_matrix = input("Save accuracy matrix for eta and lambda [Y/n] ")
    if create_accuracy_matrix == "Y" or create_accuracy_matrix == "y":
        create_accuracy_matrix = True
    elif create_accuracy_matrix == "N" or create_accuracy_matrix == "n":
        create_accuracy_matrix = False
    else:
        print("Please input Y or n!")
        sys.exit()

create_val_accuracy = input("Save validation accuracy [Y/n] ")
if create_val_accuracy == "Y" or create_val_accuracy == "y":
  create_val_accuracy = True
elif create_val_accuracy == "N" or create_val_accuracy == "n":
  create_val_accuracy = False
else:
  print("Please input Y or n!")
  sys.exit()

create_training_accuracy = input("Save training accuracy for epochs [Y/n] ")
if create_training_accuracy == "Y" or create_training_accuracy == "y":
  create_training_accuracy = True
elif create_training_accuracy == "N" or create_training_accuracy == "n":
  create_training_accuracy = False
else:
  print("Please input Y or n!")
  sys.exit()

im_shape = int(input("How big resolution? NxN: "))

if gray == False:
  data_size = (im_shape, im_shape, 3)
else:
  data_size = (im_shape,im_shape,1)

rec_field = 3
filters = 20
neuros_con = 100

if eta_lambda:
    eta = [0.01]
    lmbd = [0.0005]
else:
    eta = [0.0005,0.001,0.005,0.01]
    lmbd = [0.0001,0.0005,0.001,0.005]

epochs = 5
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
print("Unique fruits:                     ", len(paths))
print("Total fruits:                      ", tot_data)
print("Image resolution:                  %ix%i" % (im_shape, im_shape))
print("Max data set to:                   ", max_data)
print("Total runs:                        ", runs)
print("Total fruit per run:               ", lim_data)
print("Number of 3x3 kernels:             ", filters)
print("Number of neurons in hidden layer  ", neuros_con)
print("Batch size                         ", batch_size)
print("---------------------------------------------")
print()

pred = []
pred_labels = []
num_label = []
score_train = np.zeros((runs-1,2))
accuracy_map = np.zeros((len(eta),len(lmbd)))
score_val = np.zeros((runs-1,2))
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
          score_train[i] = CNN.model.evaluate(X_train,y_train)
          score_val[i] = CNN.model.evaluate(X_val, y_val, verbose=1)
          data.delete_all_data()          # clear the memory
          print("Accuracy on validation data with eta = {:.4}, lambda\
                = {:.4} is {:.4}".format(eta[j],lmbd[k],score_val[i,1]))
      trainig_time = time.time()-start_time
      """
      when all the data has been trained, test data can be run through
      the network.
      """
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

"""
Storing values necessary for plotting in CNN_plot.py
"""
          if create_confusion:
              if gray:
                  np.save("../results/plotting_data/CNN_confusion_gray",conf_matrix)
              else:
                  np.save("../results/plotting_data/CNN_confusion_color",conf_matrix)

if eta_lambda == False:
    if gray:
        np.save("../results/plotting_data/CNN_accuracy_map_gray",accuracy_map)
    else:
        np.save("../results/plotting_data/CNN_accuracy_map_color",accuracy_map)

if create_val_accuracy:
    np.save("../results/plotting_data/CNN_accuracy_validation",score_val)
    np.save("../results/plotting_data/CNN_frac_data",frac_data)
if create_training_accuracy:
    np.save("../results/plotting_data/CNN_accuracy_epoch",accuracy_epoch)
print("Time spent on training {:.4}s".format(trainig_time))

"""
The following code is only used as a tool to visualise how filters and feature maps work. The plots are put into theory section
"""


"""
import matplotlib.pyplot as plt
test_path = ["../images/test_images"]
test_label = ["apple"]

test = extract_data(test_path, test_label)

if gray: test.gray()

test.reshape(im_shape)
plt.imshow(test.data[0],cmap="viridis")
plt.title("Reshaped image (%ix%i)"%(im_shape,im_shape))

plt.tight_layout()
#plt.savefig("../results/CNN/CNN_test_analysis.pdf",bbox_inches="tight",
#                                                   pad_inches=0.1)
plt.show()


successive_outputs = [layer.output for layer in CNN.model.layers[1:]]#visualization_model = Model(img_input, successive_outputs)
visualization_model = Model(inputs = CNN.model.input, outputs = successive_outputs)#Load the input image

x = test.data / 255
successive_feature_maps = visualization_model.predict(x)# Retrieve are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in CNN.model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  print(layer_name)
  if len(feature_map.shape) == 4:
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))

    # Postprocess the feature to be visually palatable
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      # Tile each filter into a horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x# Display the grid
    scale = 20. / n_features
    if layer_name == "conv2d":
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( False )
        plt.imshow( display_grid, aspect='auto', cmap='viridis' )
#plt.savefig("../results/CNN/CNN_feature_analysis.pdf",bbox_inches="tight",
#                                                  pad_inches=0.1)
plt.show()
"""
"""
for layer in CNN.model.layers:
# check for convolutional layer
    if 'conv' not in layer.name:
        continue
# get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    plt.axis("off")
    plt.subplot(141)
    plt.imshow(filters[:,:,:,0],cmap="gist_gray")
    plt.axis("off")
    plt.subplot(142)
    plt.imshow(filters[:,:,:,1],cmap="gist_gray")
    plt.axis("off")
    plt.subplot(143)
    plt.imshow(filters[:,:,:,2],cmap="gist_gray")
    plt.axis("off")
    plt.subplot(144)
    plt.imshow(filters[:,:,:,3],cmap="gist_gray")
plt.tight_layout()
#plt.savefig("../results/CNN/CNN_filter_analysis.pdf",bbox_inches="tight",
#                                                    pad_inches=0.1)
plt.show()
#plt.figure("Reshape", figsize=(5,4))
"""
