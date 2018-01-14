import numpy as np


def cart2hom(arr):
    '''
    Convert catesian to homogenous pointd by appending a row of 1s
    '''
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shapep[1])]))


def hom2cart(arr):
    '''
    Convert homogenous to catesian by dividing each row by the last row
    '''

    # arr has shape: dimensions x num_points
    num_rows = len(arr)
    if num_rows == 1 or arr.ndim == 1:
        return arr

    return np.asarray(arr[:num_rows - 1] / arr[num_rows - 1])


def compute_fundemantal(p1, p2):
    '''
    computes the fundamental matrix from corresponding points
    (p1, p2, 3*n arrays) using the normalized 8 point algorithm.
    each row is constructed as [x’*x, x’*y, x’, y’*x, y’*y, y’, x, y, 1]
    '''

    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError("Number of points don’t match.")

    # build matrix for equations
    A = zeros((n,9))
    for i in range(n):
        A[i] = [
            p1[0,i]*p2[0,i], p1[0,i]*p2[1,i], p1[0,i]*p2[2,i],
            pp1[1,i]*p2[0,i], p1[1,i]*p2[1,i], p1[1,i]*p2[2,i],
            pp1[2,i]*p2[0,i], p1[2,i]*p2[1,i], p1[2,i]*p2[2,i],
        ]

        # compute linear least square solution
        U,S,V = np.linalg.svd(A)
        F = V[-1].reshape(3,3)

        # constrain F
        # make rank 2 by zeroing out last singular value
        U,S,V = np.linalg.svd(F)
        S[2] = 0
        F = np.dot(U, np.dot(np.diag(S), V))

        return F


def compute_epipole(F):
    '''
    Computes the (right) epipole from a fundamental matrix F.
    TODO: Compute the left epipole with F.T
    '''

    # return null space of F(Fx=0)
    U, S, V = np.linalg.svd(F)
    e = V[-1]

    return e / e[2]


def reconstruct_one_point(pt1, pt2, m1, m2):
    '''
    pt1 and m1 * X are parallel and cross product = 0
    pt1 x m1 * X = pt2 x m2 * X = 0
    '''
    A = np.vstack([
        np.dot(skew(pt1), m1),
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return p / p[3]


def reconstruct_points(p1, p2, m1, m2):
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        res[:, i] = reconstruct_one_point(p1[:, i], p2[:, i], m1, m2)

    return res
