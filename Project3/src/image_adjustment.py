import numpy as np
import matplotlib.pyplot as plt
import os
import cv2          # 'pip install opencv-python' to install this library


class extract_images():
    def __init__(self, path_to_images, labels):
        """
        path_to_images: contains a list of paths to data of interest
        labels: list of categories/labels
        Note! path_to_images and labels must be correctly ordered
        """
        self.labels = []
        self.images = []


        for path, lab in zip(path_to_images, labels):
            for i, im in enumerate(os.listdir("./"+path)):
                self.labels.append(lab)
                self.images.append(cv2.imread("./"+os.path.join(path,im)))
                if i >= 25:
                    break

        self.images = np.array(self.images)


    def reshape_images(self, img_size=10):
        """
        Adjust the size of all image,
        useful so that all inputs has same size
        """
        for i in range(len(self.images)):
            self.images[i] = cv2.resize(self.images[i], (img_size, img_size))


    def gray_images(self):
        """
        Decrease the size of data to be single
        coloured instead of RGB
        """
        for i in range(len(self.images)):
            self.images[i] = np.mean(self.images[i], axis=2)




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

    test = adjust_images(paths, labels)
    #test.reshape_images(img_size=50)
    test.gray_images()

    print(test.images.shape)
    print(test.images[0].shape)
    plt.imshow(test.images[40], cmap="gray")
    plt.show()















#
