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

    if len(contours) <= 50:
        image = cv2.bitwise_and(image, image, mask=threshold)
        print(
            '[leafscan] ' + leaf_picture_name + ' a fost procesata de leaf scan simplu')

    else:
        rect = (10, 10, width - 21, height - 21)
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, np.newaxis]
        print(
            '[leafscan] ' + leaf_picture_name + ' a fost procesata de leaf scan complex')

    image = crop_image(image, height, width)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    cv2.imwrite(os.path.join(output_folder, leaf_picture_name), image)


def crop_image(image, height, width):
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
            if image.item(i, j, 0) != 0 and image.item(i, j, 1) != 0 and image.item(i, j, 2) != 0:
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
