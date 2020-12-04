import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ExtractData():
    def __init__(self, data_dir, fruit, samples, dat_size=50):
        """
        data_dir: path to data
        fruit: fruit
        samples: number of pictures of that fruit
        dat_size: reshape image to dat_size x dat_size pixels
        """
        self.data = []
        all_files = []
        for dirpath, _, filenames in os.walk(data_dir):
            for fn in filenames:
                all_files.append(os.path.join(dirpath, fn))
        img_df = pd.DataFrame({'Filepath': all_files})
        img_df['Type'] = img_df['Filepath'].apply(lambda p: p.split(os.sep)[3])
        self.img_df = img_df[['Type', 'Filepath']]
        self.dat_size = dat_size
        self.samples = samples
        filename = []
        self.label = fruit
        if fruit not in self.img_df['Type'].unique():
            raise ValueError("Forbidden apple. Please give a valid fruit",
                             self.img_df['Type'].unique())
        for label, file in zip(self.img_df['Type'], self.img_df['Filepath']):
            if label == fruit:
                filename.append(file)
        np.random.shuffle(filename)
        filename = filename[:samples]
        for image in filename:
            img = cv2.imread(image, cv2.COLOR_BGR2RGB)
            img_GRB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(img, (dat_size, dat_size))
            self.data.append(img_resize.flatten())

    def gray_scale(self):
        """
        Make image gray_scale
        """
        dat_size = self.dat_size*self.dat_size
        data = []
        for i, image in enumerate(self.data):
            G = image[:dat_size]
            R = image[dat_size:2*dat_size]
            B = image[2*dat_size:3*dat_size]
            data.append((R + G + B)/3)
        self.data = data

    def __call__(self):
        """
        Returns images as numpy arrays in list
        """
        return self.data
