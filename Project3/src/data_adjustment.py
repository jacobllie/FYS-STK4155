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
            tot_files = len(os.listdir("./"+path))
            for i, dat in enumerate(os.listdir("./"+path)):
                self.labels.append(lab)
                self.data.append(np.array(Image.open("./"+os.path.join(path,dat))))
                #if i >= 1000:
                #    break
                print("Loading %s data: %i/%i      " % (lab, i+1, tot_files), end="\r")
        print("Loading finished!                            ")


        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.hot_vector = np.zeros((len(self.data), len(labels)))
        eye = np.eye(len(labels))
        for i in range(len(labels)):
            self.hot_vector[np.where(self.labels == labels[i])] = eye[i]



    def reshape(self, dat_size=50):
        """
        Adjust the size of all data,
        useful so that all inputs has same size
        """

        self.real_data = self.data.copy()
        print(self.data.shape)

        for i in range(len(self.data)):
            x,y,_ = self.data[i].shape
            skip_x = np.linspace(0,x-1,dat_size).astype("int")
            skip_y = np.linspace(0,y-1,dat_size).astype("int")
            self.data[i] = self.data[i][skip_x,:]
            self.data[i] = self.data[i][:,skip_y]


        new_data = np.zeros((self.data.shape + self.data[0].shape))

        for i in range(len(self.data)):
            new_data[i] = self.data[i]
        self.data = np.uint8(new_data.copy())



    def gray(self):
        """
        Decrease the size of data to be single
        coloured instead of RGB
        """
        self.data = np.mean(self.data, axis=3)




    def shuffle(self, seed=False):
        """
        shuffles data and labels equally
        """
        if seed: np.random.seed(seed)
        shuffle = np.linspace(0, len(self.data)-1, len(self.data)).astype("int")
        np.random.shuffle(shuffle)
        self.data = self.data[shuffle]
        self.real_data = self.real_data[shuffle]
        self.labels = self.labels[shuffle]
        self.hot_vector = self.hot_vector[shuffle]





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
    test.shuffle()


    print("Data shape: ", test.data.shape)
    i = 0
    plt.imshow(test.data[i], cmap="gray")
    plt.title("label = %s,  hot = %s" % (test.labels[i], test.hot_vector[i]))
    plt.show()
    #plt.close()

    """
    #test_im = np.array(Image.open("./test_images/test_apple.png"))
    test = extract_data(["/test_images"], ["apple"])
    test.reshape(20)
    print(data.data[-1:,...].shape)
    print(test.data.shape)
    """



#
