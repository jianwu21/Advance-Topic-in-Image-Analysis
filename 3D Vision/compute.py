import numpy as np


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