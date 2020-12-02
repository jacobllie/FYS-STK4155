import numpy as np
from image_adjustment import extract_images
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, regularizers, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CNN_keras:
    def __init__(self, input_shape, receptive_field, filters,
                        labels):
        """
        input_shape: the shape of input data
        receptive_field: small area connected to each neuron in next layer
        filters:
        neurons_connected:
        """
        self.labels = to_categorical(labels)
        self.model = Sequential()
        self.categories = len(labels)


    def add_layer(self):
        test = 2











if __name__ == '__main__':

    paths = ["/data_images/Apple",
             "/data_images/Banana"]
    labels = ["apple", "banana"]
