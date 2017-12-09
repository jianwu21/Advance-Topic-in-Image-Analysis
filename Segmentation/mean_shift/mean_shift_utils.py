from __future__ import division, print_function, unicode_literals

import cv2
import numpy as np

def convert_img_vec(img_rgb):
    vecs = []
    img_Luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)

    for x in range(img_Luv.shape[0]):
        for y in range(img_Luv.shape[1]):
            vecs.append(np.append([x, y], img_Luv[x, y, :]))

    return vecs


def vec2img(vec, shape):
    l, w = shape
    img_o = np.array(vec)
    img_info = img_o[:, 2:]
    c = img_info.reshape(l, w, 3)
    cc = np.array(c, dtype='uint8')
    im = cv2.cvtColor(cc, cv2.COLOR_Luv2RGB)

    return im


def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))


def neighbourhood_points(X, x_centroid, distance = 5):
    eligible_X = []
    for x in X:
        distance_between = euclid_distance(x, x_centroid)
        if distance_between <= distance:
            eligible_X.append(x)

    return eligible_X


def segmentation_kernel(Xs, bandwidths, C=1):
    # the argument bandwidths should be tuple
    # Since the kernel bandwidths is consist of hs and hr
    hs, hr = bandwidths
    weights = []
    for X in Xs:
        weight = (
            C/(hs**2 * hr**3)) * np.exp(
                -(X[0]**2 + X[1]**2)/(hs**2)) * np.exp(
                    -(X[2]**2 + X[3]**2 + X[4]**2)/(hr**2))
        weights.append(weight)

    return weights
