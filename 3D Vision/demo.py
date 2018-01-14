import cv2
from matplotlib import pyplot as plt
import numpy as np

from features import find_correspondence_points

from compute import *

im_1 = plt.imread('./rigidSfM/DSCN0976.JPG')
im_2 = plt.imread('./rigidSfM/DSCN0980.JPG')

plt.imshow(im_2)
pts1, pts2 = find_correspondence_points(im_1, im_2)

points1 = cart2hom(pts1)
points2 = cart2hom(pts2)

fig, ax = plt.subplots(1, 2)
ax[0].autoscale_view('tight')
ax[0].imshow(cv2.cvtColor(im_1, cv2.COLOR_BGR2RGB))
ax[0].plot(points1[0], points1[1], 'r.')
ax[1].autoscale_view('tight')
ax[1].imshow(cv2.cvtColor(im_2, cv2.COLOR_BGR2RGB))
ax[1].plot(points2[0], points2[1], 'r.')

# compute F
F = compute_fundamental(points1, points2)

# compute the epipole
e = compute_epipole(F)

# new figure
plt.figure()
plt.imshow(im_1)
for i in range(20):
    plot_epipolar_line(im_1, F, points2[:, i], e, False)

# new figure
plt.figure()
plt.imshow(im_2)
for i in range(20):
    plot_epipolar_line(im_2, F, points1[:, i].T, e, False)

plt.show()
