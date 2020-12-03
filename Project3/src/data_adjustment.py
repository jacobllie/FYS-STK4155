import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class extract_data():
    def __init__(self, path_to_data, labels):
        """
        path_to_data: contains a list of paths to data of interest
        labels: list of categories/labels
        Note! path_to_data and labels must be correctly ordered
        """
        self.labels = []
        self.data = []


        for path, lab in zip(path_to_data, labels):
            for i, dat in enumerate(os.listdir("./"+path)):
                self.labels.append(lab)
                self.data.append(np.array(Image.open("./"+os.path.join(path,dat))))
                if i >= 25:
                    break

        self.data = np.array(self.data)




    def reshape(self, dat_size=50):
        """
        Adjust the size of all data,
        useful so that all inputs has same size
        """
        for i in range(len(self.data)):
            x,y,_ = self.data[i].shape
            skip_x = np.linspace(0,x-1,dat_size).astype("int")
            skip_y = np.linspace(0,y-1,dat_size).astype("int")
            self.data[i] = self.data[i][skip_x,:]
            self.data[i] = self.data[i][:,skip_y]




    def gray(self):
        """
        Decrease the size of data to be single
        coloured instead of RGB
        """
        for i in range(len(self.data)):
            self.data[i] = np.mean(self.data[i], axis=2)




if __name__ == '__main__':

    #path = ["/images"]
    #lab = ["UiO"]

    #test = adjust_images(path, lab)
    #test.reshape_images(img_size=50)
    #test.gray_images()

    #print(test.images[0].shape)
    #print(test.labels)
    #plt.imshow(test.images[0])
    #plt.show()


    paths = ["/data_images/Apple",
             "/data_images/Banana"]
    labels = ["apple", "banana"]

    test = extract_data(paths, labels)
    test.reshape(dat_size=50)
    #test.gray()


    print(test.data.shape)
    print(test.data[0].shape)
    plt.imshow(test.data[0], cmap="gray")
    plt.show()







#
