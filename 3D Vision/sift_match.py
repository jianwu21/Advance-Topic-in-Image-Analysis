import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_correspondence_points(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    # Find point matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Applying Lowe's SIFT matching ratio test to detect whether the
    # correspondence is true or not.
    good = []
    bad = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append([m])
        else:
            bad.append([m])

    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:200],
    #                           None, matchColor=(0,255,0), flags=2)

    # save the out_put image for displaying
    # plt.imsave('./imgs/feature_match.jpg', img3)
    # plt.imshow(img3)

    src_pts = np.asarray([kp1[m[0].queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m[0].trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return src_pts.T, dst_pts.T
