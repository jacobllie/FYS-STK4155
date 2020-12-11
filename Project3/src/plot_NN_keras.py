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

plt.figure(figsize=(5,4))
for key, label in zip(DATA.keys()[1:4], labels):
    plt.plot(im_load[:-1], DATA[key].values[:-1], label=label)
plt.xlabel("Images trained")
plt.ylabel("Accuracy")
plt.title("Accurracy on validation data from FFNN")
plt.legend()
plt.tight_layout()
plt.savefig("../results/NN/FFNN_accuracy.pdf",
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()


plt.figure(figsize=(5,4))
for key, label in zip(DATA.keys()[4:], labels):
    plt.plot(im_load[:-1], DATA[key].values[:-1], label=label)
plt.xlabel("Images trained")
plt.ylabel("Accuracy")
plt.title("Accurracy on validation data from Keras")
plt.legend()
plt.tight_layout()
plt.savefig("../results/Keras/keras_accuracy.pdf",
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()
