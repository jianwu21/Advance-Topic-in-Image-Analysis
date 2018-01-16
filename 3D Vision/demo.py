import cv2
from matplotlib import pyplot as plt
import numpy as np

from compute import *
from features import find_correspondence_points
from FmatrixModel import FundamentalMatrixModel

im_1 = plt.imread('./rigidSfM/DSCN0976.JPG')
im_2 = plt.imread('./rigidSfM/DSCN0972.JPG')

pts1, pts2 = find_correspondence_points(im_1, im_2)

num = pts1.shape[1]
print('Totally {} points correspondence'.format(num))

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
model = FundamentalMatrixModel()
F = model.fit(points1, points2)
print(model.error(F, points1, points2))

# compute the epipole
e = compute_epipole(F)

# plot_epipolar_lines(im_1, im_2, points1, points2, F, show_epipole=True)
plt.figure()
plt.imshow(im_1)
plt.plot(points1[0], points1[1], 'r.')
for i in range(points1.shape[1]):
    plot_epipolar_line(im_1, F, points2[:, i], epipole=None, show_epipole=True)

plt.figure()
plt.imshow(im_2)
plt.plot(points2[0], points2[1], 'r.')
for i in range(points1.shape[1]):
    plot_epipolar_line(im_2, F.T, points1[:, i], epipole=None, show_epipole=True)

plt.show()
