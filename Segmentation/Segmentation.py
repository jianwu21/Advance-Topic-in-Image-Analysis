# coding: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage.transform import resize

from mean_shift.mean_shift_utils import *
from mean_shift.mean_shift import *

from mean_shift.point_grouper import *


def demo():
    orig_img = plt.imread('data/DSC_0055.JPG')[1000:2000, 1600:2600]
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.axis('off')

    resized_img = 255 * resize(
        orig_img,
        (orig_img.shape[0] / 10, orig_img.shape[1] / 10),
    )
    img = resized_img.astype(np.uint8)

    plt.subplot(1, 2, 2)
    plt.imshow(img)

    plt.axis('off')

    plt.imsave('./orig.png', img)

    # Mean Shift Filter
    vecs = convert_img_vec(img)

    plt.imshow(vec2img(vecs, img.shape[:2]))
    plt.title('Original')
    plt.axis('off')

    bandwidths_list = [
        (8, 8), (8, 16),
        (16, 4), (16, 8), (16, 16),
        (32, 4), (32, 8), (32, 16),
    ]
    results = {}

    for bandwidths in bandwidths_list:
        ms = mean_shift(kernel=segmentation_kernel)
        vec_c = ms.cluster(vecs, kernel_bandwidth=bandwidths)
        shift_im = vec2img(vec_c, img.shape[:2])
        results[bandwidths] = shift_im

    for key in bandwidths_list:
        value = results[key]
        hs, hr = key
        plt.figure()
        plt.axis('off')
        plt.title('hs={}, hr={}'.format(hs, hr))
        plt.imshow(value)
        plt.imsave('./{}_{}.png'.format(hs, hr), value)

    vec_c = results[(32, 16)]
    clf = PointGrouper()
    groups = clf.group_points(vec_c)

    plt.show()


if __name__ == '__main__':
    demo()
