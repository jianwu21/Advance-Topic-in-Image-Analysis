import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

from compute import *
from sift_match import find_correspondence_points
from FmatrixModel import FundamentalMatrixModel
from OptimizeFmatrix import *
from ransac import *


def demo(path_1, path_2):
    im_1 = plt.imread(path_1)
    im_2 = plt.imread(path_2)

    pts1, pts2 = find_correspondence_points(im_1, im_2)

    num = pts1.shape[1]
    print('Totally {} points correspondence'.format(num))

    points1 = cart2hom(pts1)
    points2 = cart2hom(pts2)

    plt.figure()
    plt.imshow(cv2.cvtColor(im_1, cv2.COLOR_BGR2RGB))
    plt.plot(points1[0], points1[1], 'r.')

    plt.figure()
    plt.imshow(cv2.cvtColor(im_2, cv2.COLOR_BGR2RGB))
    plt.plot(points2[0], points2[1], 'r.')

    # original F without RANSAC
    model = FundamentalMatrixModel()
    F_bad = model.fit(points1, points2)
    plot_epipolar_lines(
        im_1,
        im_2,
        points1,
        points2,
        F_bad,
        show_epipole=False)

    # opt
    F0, F, inliers = fmatrix(points1.T, points2.T)
    print(inliers)
    plot_epipolar_lines(
        im_1, im_2, points1[:, inliers], points2[:, inliers], F0,
        show_epipole=False)
    plot_epipolar_lines(
        im_1, im_2, points1[:, inliers], points2[:, inliers], F,
        show_epipole=False)

    # Using cv2 to find
    cv_F, mask = cv2.findFundamentalMat(
        points1[:2].T, points2[:2].T, cv2.FM_RANSAC)
    print('The number of inliner by OpenCV is {}'.format(len(mask.ravel()==1)))
    plot_epipolar_lines(
        im_1,
        im_2,
        points1[:, inliers],
        points2[:, inliers],
        cv_F,
        show_epipole=False)

    plt.show()


if __name__ == '__main__':
    demo(sys.argv[1], sys.argv[2])
