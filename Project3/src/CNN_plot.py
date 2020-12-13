import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import os


path = "../results/plotting_data/"
plt.rcParams["figure.figsize"] = (5,4)


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

    gray = input("[gray/color]?: ")
    if gray == "gray":
      gray = True
    elif gray == "color":
      gray = False
    else:
      print("Please input gray or color!")
      sys.exit()

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

    if gray: heatmap.set_title("Confusion matrix for fruit recognition (Gray)")
    else: heatmap.set_title("Confusion matrix for fruit recognition (Color)")
    fig = heatmap.get_figure()

    plt.show()

    save_fig = input("Save figure [Y/n]? ")
    if save_fig=="Y" or save_fig=="y":
        name = input("Saving file as *.pdf file. Please enter name of file: ")
        fig.savefig("../results/CNN/%s.pdf" % name, pad_inches=0.1)
        print("File saved successfully in the ../results/CNN/ folder!")




"""
Creating an accuracy map
"""

create_accuracy_matrix = input("Analyse accuracy map [Y/n]: ")

if create_accuracy_matrix == "Y" or create_accuracy_matrix == "y":
    create_accuracy_matrix = True
    print("These are the current files available for making an accuracy map:")
    i = 0
    path_list = []
    for acc_path in os.listdir(path):
        if acc_path[:3]=="acc":
            print(i, acc_path)
            path_list.append(acc_path)
            i += 1
    num = input("Please choose the file to plot by entering the corresponding integer: ")
    try:
        acc_plot = np.load(path+path_list[int(num)])
    except:
        print("'%s' not a valid integer! Please try again. " % num)
        sys.exit()

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
    elif "Epochs" in path_list[int(num)] and "Batch" in path_list[int(num)]:
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

    if "gray" in path_list[int(num)]:
        col = "Gray"
    elif "RGB" in path_list[int(num)]:
        col = "Color"

    heatmap.set_title("Accuracy map for fruit recognition (%s)" % col)
    fig = heatmap.get_figure()
    plt.tight_layout()
    plt.show()
    save_fig = input("Save figure [Y/n]? ")
    if save_fig=="Y" or save_fig=="y":
        name = input("Saving file as *.pdf file. Please enter name of file: ")
        fig.savefig("../results/CNN/%s.pdf" % name, pad_inches=0.1)
        print("File saved successfully in the ../results/CNN/ folder!")




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

    fig,ax = plt.subplots()
    plt.style.use("seaborn")
    ax.set_title("Accuracy as a function of runs")
    ax.set_xlabel("% of data used for training")
    ax.set_ylabel("Accuracy")
    ax.plot(frac_data,accuracy_val[:,1],"purple",label="Validation")
    ax.legend()
    fig.show()

    save_fig = input("Save figure [Y/n]? ")
    if save_fig=="Y" or save_fig=="y":
        name = input("Saving file as *.pdf file. Please enter name of file: ")
        fig.savefig("../results/CNN/%s.pdf" % name, pad_inches=0.1)
        print("File saved successfully in the ../results/CNN/ folder!")



"""
Creating a epoch accuracy plot
"""

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

    fig,ax = plt.subplots()
    plt.style.use("seaborn")
    ax.set_title("Accuracy of training data at different stages of training")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    for i in range(len(accuracy_epoch)):
        ax.plot(np.arange(accuracy_epoch.shape[1]),accuracy_epoch[i,:],label="{:.3}%".format(frac_data[i]))
    ax.legend()
    fig.show()

    save_fig = input("Save figure [Y/n]? ")
    if save_fig=="Y" or save_fig=="y":
        name = input("Saving file as *.pdf file. Please enter name of file: ")
        fig.savefig("../results/CNN/%s.pdf" % name, pad_inches=0.1)
        print("File saved successfully in the ../results/CNN/ folder!")
