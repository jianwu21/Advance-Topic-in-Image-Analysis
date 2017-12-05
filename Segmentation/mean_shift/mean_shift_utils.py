def convert_img_vec(img_rgb):
    vecs = []
    img_Luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)

    for x in range(img_Luv.shape[0]):
        for y in range(img_Luv.shape[1]):
            vecs.append(np.append([x, y], img_Luv[x, y, :]))

    return vecs


def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))


def neighbourhood_points(X, x_centroid, distance = 5):
    eligible_X = []
    for x in X:
        distance_between = euclid_distance(x, x_centroid)
        if distance_between <= distance:
            eligible_X.append(x)

    return eligible_X


def segmentation_kernel(hs, hr, C, X):
    val = (C/(hs**2 * hr**3)) * np.exp(-(X[0]**2 + X[1]**2)/(hs**2)) * np.exp(-(X[2]**2 + X[3]**2 + X[4]**2)/(hr**2))()

    return val
