import cv2
from matplotlib import pyplot as plt
import numpy as np

from features import find_correspondence_points

im_1 = plt.imread('./rigidSfM/DSCN0975.JPG')
im_2 = plt.imread('./rigidSfM/DSCN0974.JPG')

plt.imshow(im_2)
i, j = find_correspondence_points(im_1, im_2)
