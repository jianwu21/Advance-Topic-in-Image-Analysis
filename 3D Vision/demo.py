import cv2
from matplotlib import pyplot as plt
import numpy as np

from compute import *
from sift_match import find_correspondence_points
from FmatrixModel import FundamentalMatrixModel
from OptimizeFmatrix import *
from ransac import *


def demo():
    im_1 = plt.imread('./rigidSfM/DSCN0974.JPG')
    im_2 = plt.imread('./rigidSfM/DSCN0975.JPG')

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

    '''
    # Using cv2 to find
    F, mask = cv2.findFundamentalMat(
        points1[:2].T, points2[:2].T, cv2.FM_RANSAC)
    print('The number of inliner by OpenCV is {}'.format(len(mask.ravel()==1)))
    plot_epipolar_lines(
        im_1, im_2, points1[:, mask.ravel()==1], points2[:, mask.ravel()==1], F, show_epipole=False)
    '''
    # opt
    F0, F, inliers = fmatrix(points1.T, points2.T)
    print(inliers)
    plot_epipolar_lines(
        im_1, im_2, points1[:, inliers], points2[:, inliers], F0,
        show_epipole=False)
    plot_epipolar_lines(
        im_1, im_2, points1[:, inliers], points2[:, inliers], F,
        show_epipole=False)

    plt.show()

if __name__ == '__main__':
    demo()
