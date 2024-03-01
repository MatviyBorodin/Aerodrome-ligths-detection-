import cv2
import numpy as np
import matplotlib.pyplot as plt 
from skimage import io, measure


img = io.imread('E:\Programming\PythonProjects\PythonFiles\YELLOW.png')

lower = np.array([29, 200, 200])
upper = np.array([31, 255, 255])

hsv = cv2.  cvtColor(img, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv, lower, upper)


from scipy import ndimage as nd
closed_mask = nd.binary_closing(mask, np.ones((7, 7)))


label_image = measure.label(closed_mask)
plt.imshow(label_image)

from skimage.color import label2rgb
image_label_overlay = label2rgb(label_image, image = img)
plt.imshow(image_label_overlay)

props = measure.regionprops_table(label_image, img,
                                  properties=['label', 
                                              'area', 'equivalent_diameter',
                                              'solidity'])

import pandas as pd
df = pd.DataFrame(props)
print(df.head())

