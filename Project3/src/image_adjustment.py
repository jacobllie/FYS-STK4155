import numpy as np
import matplotlib.pyplot as plt
import os
import cv2          # 'pip install opencv-python' to install this library


class adjust_images():
    def __init__(self, path_to_images, labels):
        """
        path_to_images: list of paths
        labels: list of categories/labels
        """
        self.labels = []
        self.images = []


        for path, lab in zip(path_to_images, labels):
            for im in os.listdir("./"+path):
                self.labels.append(lab)
                self.images.append(cv2.imread("./"+os.path.join(path,im)))

        self.images = np.array(self.images)


    def reshape_images(self, img_size=10):
        for i in range(len(self.images)):
            self.images[i] = cv2.resize(self.images[i], (img_size, img_size))


    def gray_images(self):
        for i in range(len(self.images)):
            self.images[i] = np.mean(self.images[i], axis=2)






path = ["/images"]
lab = ["UiO"]

test = adjust_images(path, lab)
test.reshape_images(img_size=50)
test.gray_images()

print(test.images[0].shape)
print(test.labels)
plt.imshow(test.images[0])
plt.show()
