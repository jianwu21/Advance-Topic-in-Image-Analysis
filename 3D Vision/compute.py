from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt


def cart2hom(arr):
    '''
    Convert catesian to homogenous pointd by appending a row of 1s
    '''
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))


def hom2cart(arr):
    '''
    Convert homogenous to catesian by dividing each row by the last row
    '''

    # arr has shape: dimensions x num_points
    num_rows = len(arr)
    if num_rows == 1 or arr.ndim == 1:
        return arr

    return np.asarray(arr[:num_rows - 1] / arr[num_rows - 1])


def compute_epipole(F):
    '''
    Computes the (right) epipole from a fundamental matrix F.
    TODO: Compute the left epipole with F.T
    '''

    # return null space of F(Fx=0)
    U, S, V = np.linalg.svd(F)
    e = V[-1]

    return e / e[2]


def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
    '''
    Plot the epipole and epipolar line F*x=0
    in an image. F is the fundamental matrix
    and x a point in the other image.
    '''

    m, n = im.shape[:2]
    line = np.dot(F, x)
    # epipolar line parameter and values
    t = np.linspace(0, n, 100)
    lt = np.array(
        [(line[2]+line[0]*tt)/(-line[1]) for tt in t]
    )
    # take only line points inside the image
    ndx = (lt>=0) & (lt<m)
    plt.plot(t[ndx], lt[ndx], linewidth=2)
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0]/epipole[2], epipole[1]/epipole[2], 'r*')


def plot_epipolar_lines(im_1, im_2, points1, points2, F, show_epipole=False):
    ''' Plot the points and epipolar lines. P1' F P2 = 0 '''
    num = points1.shape[1]

    # show the epipolar line for im_1
    plt.figure()
    plt.imshow(im_1)

    for i in range(num):
        plot_epipolar_line(im_1, F, points2[:, i], epipole=None, show_epipole=True)

    # show the epipolar line for im_2
    plt.figure()
    plt.imshow(im_2)

    for i in range(num):
        plot_epipolar_line(im_2, F.T, points1[:, i], epipole=None, show_epipole=True)
