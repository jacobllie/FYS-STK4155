"""
Script for plotting accuracy of FFNN from our own produced code and Keras
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn")
plt.style.use({'font.size': 18})

REL_PATH = "../results/"
FILE_NAME = "NN-keras_data.csv"

DATA = pd.read_csv(REL_PATH + FILE_NAME)

runs = DATA["runs"].values
im_load = 71*runs*0.8

labels = ["Color 50x50", "B/w 50x50", "B/w 85x85"]

fig,ax = plt.subplots(figsize=(5,4))
for key, label in zip(DATA.keys()[1:4], labels):
    ax.plot(im_load[:-1], DATA[key].values[:-1], label=label)
ax.set_xlabel("Images trained")
ax.set_ylabel("Accuracy")
ax.set_title("Accurracy on validation data from FFNN")
ax.legend()
fig.tight_layout()
fig.show()

save_fig = input("Save figure [Y/n]? ")
if save_fig=="Y" or save_fig=="y":
    name = input("Saving file as *.pdf file. Please enter name of file: ")
    fig.savefig("../results/CNN/%s.pdf" % name,
                           bbox_inches='tight',
                           pad_inches=0.1)
    print("File saved successfully in the ../results/NN/ folder!")



fig,ax = plt.subplots(figsize=(5,4))
for key, label in zip(DATA.keys()[4:], labels):
    plt.plot(im_load[:-1], DATA[key].values[:-1], label=label)
ax.set_xlabel("Images trained")
ax.set_ylabel("Accuracy")
ax.set_title("Accurracy on validation data from Keras")
ax.legend()
fig.tight_layout()
fig.show()

save_fig = input("Save figure [Y/n]? ")
if save_fig=="Y" or save_fig=="y":
    name = input("Saving file as *.pdf file. Please enter name of file: ")
    fig.savefig("../results/CNN/%s.pdf" % name,
                           bbox_inches='tight',
                           pad_inches=0.1)
    print("File saved successfully in the ../results/Keras/ folder!")
