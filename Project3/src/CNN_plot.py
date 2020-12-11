import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import os


gray = input("gray/color?: ")
if gray == "gray":
  gray = True
elif gray == "color":
  gray = False
else:
  print("Please input gray or color!")
  sys.exit()

path = "../plotting_data/"




"""
Creating a confusion matrix
"""

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
    heatmap.set_xlabel("Prediction")
    heatmap.set_ylabel("True labelling")

    heatmap.set_title("Confusion matrix for fruit recognition (Gray)")
    fig = heatmap.get_figure()
    #fig.savefig("../figures/CNN_confusion_gray.pdf", bbox_inches='tight',
    #                                            pad_inches=0.1)
    plt.show()





"""
Creating an accuracy map
"""

create_accuracy_matrix = input("Analyse accuracy map [Y/n]: ")

if create_accuracy_matrix == "Y" or create_accuracy_matrix == "y":
    create_accuracy_matrix = True
    print("These are the current files available for making an accuracy map:")
    i = 0
    path_list = []
    for path in os.listdir("../results/plotting_data/"):
        if path[:3]=="acc":
            print(i, path)
            path_list.append(path)
            i += 1
    num = input("Please choose the file to plot by entering the corresponding integer: ")
    try:
        acc_plot = np.load("../results/plotting_data/"+path_list[int(num)])
    except:
        assert False, "'%s' not a valid integer! Please try again. " % num

elif create_accuracy_matrix == "N" or create_accuracy_matrix == "n":
    create_accuracy_matrix = False
else:
    print("Please input Y or n!")
    sys.exit()


if create_accuracy_matrix == True:
    if "eta" in path_list[int(num)] and "lambda" in path_list[int(num)]:
        param1 = [0.0005,0.001,0.005,0.01]         # eta values
        param2 = [0.0001,0.0005,0.001,0.005]      # lambda values
        param_name = [r"$\eta$", r"$\lambda$"]
    elif "epoch" in path_list[int(num)] and "batch" in path_list[int(num)]:
        param1 = [1, 3, 5, 10]                 # epoch values
        param2 = [1, 3, 5, 10]             # batch size values
        param_name = ["Epochs", "Batch size"]


    heatmap = sb.heatmap(acc_plot,cmap="viridis",
                          xticklabels=["%s" %i for i in param2],
                          yticklabels=["%s" %i for i in param1],
                          cbar_kws={'label': 'Accuracy'},
                          fmt = ".4",
                          edgecolor="none",
                          annot = True)
    plt.yticks(rotation=0)
    heatmap.set_xlabel(param_name[1])
    heatmap.set_ylabel(param_name[0])


    heatmap.set_title("Accuracy map for fruit recognition")
    fig = heatmap.get_figure()
    plt.tight_layout()
    plt.show()
    save_fig = input("Save figure [Y/n]? ")
    if save_fig=="Y" or save_fig=="y":
        name = input("Saving file as *.pdf file. Please enter name of file: ")
        fig.savefig("../figures/%s.pdf" % name, pad_inches=0.1)
        print("File saved successfully in the ../figures/ folder!")




"""
Creating a validation accuracy plot
"""

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
