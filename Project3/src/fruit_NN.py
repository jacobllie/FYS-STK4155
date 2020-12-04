import os
import cv2

import Network as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from extract_fruit import ExtractData

path = "../data/fruit"

apples = ExtractData(path, "Apple", 100)
apples.gray_scale()
Apple = apples()

bananas = ExtractData(path, "Banana", 100)
bananas.gray_scale()
Banana = bananas()

fruit_list = np.concatenate([Apple, Banana])
