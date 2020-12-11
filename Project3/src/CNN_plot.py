import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sys

gray = input("gray/color?: ")
if gray == "gray":
  gray = True
elif gray == "color":
  gray = False
else:
  print("Please input gray or color!")
  sys.exit()

path = "../plotting_data/"


create_conf_matrix = input("Analyse confusion matrix [Y/n]: ")

if create_conf_matrix == "Y" or create_conf_matrix == "y":
    create_conf_matrix = True
elif create_conf_matrix == "N" or create_conf_matrix == "n":
    create_conf_matrix = False
else:
    print("Please input Y or n!")
    sys.exit()
if create_conf_matrix == True:
    true_labels = ["apple", "banana", "kiwi", "mango",
                   "orange", "pear", "tomato"]
    if gray:
        conf_matrix = np.load(path+"CNN_confusion_gray.npy")
    else:
        conf_matrix = np.load(path+"CNN_confusion_color.npy")
    heatmap = sb.heatmap(conf_matrix,cmap="viridis",
                          xticklabels=["%s" %i for i in true_labels],
                          yticklabels=["%s" %i for i in true_labels],
                          cbar_kws={'label': 'Accuracy'},
                          fmt = ".3f",
                          edgecolor="none",
                          annot = True)
    plt.yticks(rotation=0)
    heatmap.set_xlabel("pred")
    heatmap.set_ylabel("true")

    heatmap.set_title("Confusion matrix for fruit recognition (Gray)")
    fig = heatmap.get_figure()
    #fig.savefig("../figures/CNN_confusion_gray.pdf", bbox_inches='tight',
    #                                            pad_inches=0.1)
    plt.show()

create_accuracy_matrix = input("Analyse accuracy map [Y/n]: ")


if create_accuracy_matrix == "Y" or create_accuracy_matrix == "y":
    create_accuracy_matrix = True
elif create_accuracy_matrix == "N" or create_accuracy_matrix == "n":
    create_accuracy_matrix = False
else:
    print("Please input Y or n!")
    sys.exit()
if create_accuracy_matrix == True:
    eta = [0.0005,0.001,0.005,0.01]
    lmbd = [0.0001,0.0005,0.001,0.005]
    if gray:
        accuracy_map = np.load(path+"CNN_accuracy_map_gray.npy")
    else:
        accuracy_map =np.load(path+"CNN_accuracy_map_color.npy")

    heatmap = sb.heatmap(accuracy_map.T,cmap="viridis",
                          xticklabels=["%.4f" %i for i in eta],
                          yticklabels=["%.4f" %i for i in lmbd],
                          cbar_kws={'label': 'Accuracy'},
                          fmt = ".4",
                          edgecolor="none",
                          annot = True)
    plt.yticks(rotation=0)
    heatmap.set_xlabel(r"$\eta$")
    heatmap.set_ylabel(r"$\lambda$")

    heatmap.set_title("Accuracy map for fruit recognition")
    fig = heatmap.get_figure()
    #fig.savefig("../figures/CNN_accuracy_map.pdf", bbox_inches='tight',
                                                   #pad_inches=0.1)
    plt.show()


create_val_accuracy = input("Analyse validation accuracy as a function of data trained [Y/n]: ")

if create_val_accuracy == "Y" or create_val_accuracy == "y":
    create_val_accuracy = True
elif create_val_accuracy == "N" or create_val_accuracy == "n":
    create_val_accuracy= False
else:
    print("Please input Y or n!")
    sys.exit()
if create_val_accuracy == True:

    accuracy_val = np.load(path+"CNN_accuracy_validation.npy")
    frac_data = np.load(path+"CNN_frac_data.npy")

    plt.style.use("seaborn")
    plt.title("Accuracy as a function of runs")
    plt.xlabel("% of data used for training")
    plt.ylabel("Accuracy")
    plt.plot(frac_data,accuracy_val[:,1],"purple",label="Validation")
    plt.legend()
    #plt.savefig("../figures/CNN_val_acc_layers.pdf",bbox_inches="tight",
    #                                                 pad_inches=0.1)
    plt.show()

create_epoch_accuracy = input("Analyse accuracy of data at different % of data trained [Y/n]: ")

if create_epoch_accuracy == "Y" or create_epoch_accuracy == "y":
    create_epoch_accuracy = True
elif create_epoch_accuracy == "N" or create_epoch_accuracy == "n":
    create_epoch_accuracy = False
else:
    print("Please input Y or n!")
    sys.exit()
if create_epoch_accuracy == True:
    frac_data = np.load(path+"CNN_frac_data.npy")
    accuracy_epoch = np.load(path+"CNN_accuracy_epoch.npy")
    print(accuracy_epoch)
    plt.style.use("seaborn")
    plt.title("Accuracy of training data at different stages of training")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    for i in range(len(accuracy_epoch)):
        plt.plot(np.arange(accuracy_epoch.shape[1]),accuracy_epoch[i,:],label="{:.3}%".format(frac_data[i]))
    plt.legend()
    #plt.savefig("../figures/CNN_acc_training.pdf",bbox_inches="tight",
    #                                              pad_inches=0.1)
    plt.show()
