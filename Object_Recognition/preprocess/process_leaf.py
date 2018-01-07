from __future__ import absolute_import, division, print_function

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def process_scan_leaf(scan_leaf_path, output_folder):
    leaf_picture_name = os.path.split(scan_leaf_path)[-1]

    image = cv2.imread(scan_leaf_path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_image = cv2.GaussianBlur(grey_image, (5, 5), 3)
    ret, threshold = cv2.threshold(
        grey_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy, _ = cv2.findContours(
        threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    height, width, depth = image.shape

    for i in range(height-1):
        for j in range(width-1):
            if contours[i, j] == 0:
                image[i, j] = np.array([255, 255, 255])

    print(
        '[leafscan] ' + leaf_picture_name + \
        ' a fost procesata de leaf scan complex')

    image = crop_image(image)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    cv2.imwrite(os.path.join(output_folder, leaf_picture_name), image)


def crop_image(image):
    # set height and width
    height, width = image.shape[:2]

    top = height - 1
    bottom = 0
    left = width - 1
    right = 0

    print(
        '[crop] Top: {} Bottom: {} Left: {} Right: {}'.format(
            top, bottom, left, right)
    )

    for i in range(height):
        for j in range(width):
            if image.item(i, j, 0) != 255 \
            and image.item(i, j, 1) != 255 \
            and image.item(i, j, 2) != 255:
                if i < top:
                    top = i
                elif i > bottom:
                    bottom = i

                if j < left:
                    left = j
                elif j > right:
                    right = j

    print('[crop] Top: {} Bottom: {} Left: {} Right: {}'.format(
        top, bottom, left, right
    ))

    image = image[top:bottom, left:right]

    return image
